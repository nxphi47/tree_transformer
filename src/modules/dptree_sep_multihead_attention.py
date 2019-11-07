import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules.multihead_attention import *

DEBUG = False


def maybe_print(s):
    if DEBUG:
        print('dptree_step_multihead_attention.py::' + s)


class DPTreeSeparateOnlyKeyAttention(nn.Module):
    def __init__(self, args, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.src_len_norm = getattr(args, 'src_len_norm', 'sqrt')

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def norm_src_len(self, src_lens_denom):
        if self.src_len_norm == 'sqrt':
            return src_lens_denom.sqrt_()
        elif self.src_len_norm == 'none':
            return torch.tensor(1.0, device=src_lens_denom.device, dtype=src_lens_denom.dtype)
        elif self.src_len_norm == 'intact':
            return src_lens_denom
        else:
            raise NotImplementedError(f'{self.src_len_norm}')

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )

    def prepare_dptree_qkv(self, query, key, value, key_padding_mask=None, incremental_state=None,
                           need_weights=True, static_kv=False, force_self_att=False):

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, query_bsz, embed_dim = query.size()
        ori_src_len, key_bsz, embed_dim_ = key.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, query_bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same or force_self_att:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
            # raise NotImplementedError(f'encoder-decoder attention not yet')
        else:
            # q = self.in_proj_q(query)
            # k = self.in_proj_k(key)
            # v = self.in_proj_v(value)
            raise NotImplementedError(f'free-style attention not yet')
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            raise NotImplementedError

        q = q.contiguous().view(tgt_len, query_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            assert static_kv, f'static_kv={static_kv}, only cross-attention impl here'
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(key_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(key_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(key_bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(key_bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            # assert key_padding_mask.size(0) == key_bsz
            assert key_padding_mask.size(-1) == src_len

        if self.add_zero_attn:
            raise NotImplementedError

        return q, k, v, key_padding_mask, saved_state, src_len, tgt_len, query_bsz

    @classmethod
    def scores_affinity2dptable(cls, affinity, flat_indices, seq_len):
        """

        :param affinity:        [b, m, tq, tk]
        :param flat_indices:    [b, m, tq, tk]
        :param seq_len:
        :return:
        """
        size = affinity.size()
        bsz, aff_nsent, tq, _tk = affinity.size()
        batch, nsent, tq, __tk = flat_indices.size()
        assert bsz == batch, f'seq_size {size}, flidx_size {flat_indices.size()}'
        assert aff_nsent == nsent, f'{affinity.size()} ??? {flat_indices.size()}'

        zeros = torch.zeros(batch, nsent, tq, seq_len * seq_len, dtype=affinity.dtype, device=affinity.device)
        # zeros:        [b, m, tq, T**2]
        matrix = zeros.scatter_(dim=3, index=flat_indices, src=affinity)
        # matrix:       [b, m, tq, T ** 2]
        matrix = matrix.view(batch, nsent, tq, seq_len, seq_len)
        # matrix:   [B, m, tq, seq_len, seq_len]
        return matrix

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len):
        """

        :param q:           [bq * h, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, tq, d = q.size()
        q = q.unsqueeze(1)
        # q:                [bq * h, 1, tq, d]
        linear_scores = torch.matmul(q, k.transpose(2, 3))

        matrix_scores = self.__class__.scores_affinity2dptable(linear_scores, fl_idx, seq_len)
        # matrix:           [b, * h, m, tq, t, t]
        """
        a b c d
        0 f g h
        0 0 i j
        0 0 0 k
        
        a 0 0 0
        b f 0 0
        c g i 0
        d h j k
        """
        acc_fw_matrix = torch.cumsum(matrix_scores, dim=4)
        acc_bw_matrix = torch.cumsum(matrix_scores, dim=3).transpose(3, 4)
        dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert tq == tq_dp_mat, f'{dp_matrix.size()} ??? {q.size()} ??? {linear_scores.size()} ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, tq, t1 * t2)
        dp_scores = torch.gather(dp_linear_mat, dim=3, index=gt_idx)

        return dp_scores

    @classmethod
    def indices2gt_indices(cls, indices, seq_len, query_len, heads, nsent, head_dim=None):
        """

        :param indices:     [b, m, tk, 2]
        :param seq_len:
        :param query_len:
        :param heads:
        :param head_dim:
        :return:    [B * h, m, tq, Tk]
        """
        gt_idx = (indices[:, :, :, 0] * seq_len + indices[:, :, :, 1]).unsqueeze_(1).unsqueeze_(3)
        # size = gt_idx.size()
        bsz, _, nsent, _, tk = gt_idx.size()
        # gt_idx = gt_idx.expand(-1, query_len * heads, -1).contiguous().view(size[0] * heads, query_len, size[-1])
        # gt_idx = gt_idx.expand(-1, heads, nsent, query_len, -1).contiguous().view(bsz * heads, nsent, query_len, tk)
        gt_idx = gt_idx.expand(-1, heads, -1, query_len, -1).contiguous().view(bsz * heads, nsent, query_len, tk)
        return gt_idx

    @classmethod
    def indices2flat_indices(cls, indices, seq_len, head_dim, heads, nsent, query_len=None):
        assert query_len is not None
        fl_idx = cls.indices2gt_indices(indices, seq_len, query_len, heads, nsent=nsent)
        return fl_idx

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, tq, d]
        :param k:                   [B * h, m, tk, d]
        :param v:                   [B * h, m, tk, d] ---> [b * h, tk * m, d] ?
        :param fl_idx:              [B * h, m, tq, Tk] nor [B * h, d, Tk]
        :param gt_idx:              [B * h, m, tq, tk]
        :param attn_mask:
        :param key_padding_mask:    [B, m, tk]
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        k_size = k.size()
        node_len = k_size[2]
        nsent = k_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        attn_weights = self.dptree_dot_product(q, k, fl_idx, gt_idx, seq_len)
        assert not torch.isnan(attn_weights).any()

        # assert list(attn_weights.size()) == [bsz * self.num_heads, nsent, tgt_len, src_len],
        if list(attn_weights.size()) != [bsz * self.num_heads, nsent, tgt_len, src_len]:
            raise ValueError(f'{attn_weights.size()} != {[bsz * self.num_heads, nsent, tgt_len, src_len]}, q={q.size()}, fl_idx={fl_idx.size()}')

        if attn_mask is not None:
            raise NotImplementedError('attn_mask for decoder not yet')

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            src_lens_denom = (~exp_pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

            attn_weights = attn_weights.float().masked_fill(
                exp_pad_mask, float('-inf')).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

        assert not torch.isnan(attn_weights).any(), f'src_lens: {src_lens_denom is None}: {src_lens_denom == 0}'

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)
        # attn_weights[torch.isnan(attn_weights)] = 0.0

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights:     [b * h, m, tgt_len, src_len]
        attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous().view(bsz * self.num_heads, tgt_len, src_len * nsent)
        # attn_weights:     [b * h, tgt_len, src_len * m]
        assert not torch.isnan(attn_weights).any()

        attn = torch.bmm(attn_weights, v)
        assert not torch.isnan(attn).any()

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        assert not self.onnx_trace
        # if (self.onnx_trace and attn.size(1) == 1):
        #     attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        # else:
        #     attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len * nsent)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None
        return attn, attn_weights

    def forward(self, query, key, value, flat_indices, gt_indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, force_self_att=False):
        """Input shape: Time x Batch x Channel

        :param query:           [Tq, B, C]
        :param key:             [Tk, B, m, C]
        :param value:           [Tk, B, m, C]
        :param flat_indices:    [B * h, m, Tq, Tk]
        :param gt_indices:      [B * h, m, Tq, Tk]
        :param key_padding_mask:    [B, m, Tk]
        """

        assert flat_indices.size() == gt_indices.size(), f'{flat_indices.size()} != {gt_indices.size()}'
        tq, query_bsz, dim = query.size()
        tk, key_bsz, nsent, dim_k = key.size()
        assert query_bsz == key_bsz

        assert attn_mask is None, f'not None attn_mask (decoder self-attention) not ready'

        f_key = key.view(tk, key_bsz * nsent, dim_k)
        f_value = value.view(tk, key_bsz * nsent, dim_k)
        # f_key_pad_mask = key_padding_mask = key_padding_mask.view(key_bsz * nsent, tk)
        assert not torch.isnan(query).any()
        (q, f_key, f_value, key_padding_mask, saved_state, src_len, tgt_len, query_bsz_) = self.prepare_dptree_qkv(
            query, f_key, f_value, key_padding_mask, incremental_state, need_weights, static_kv,
            force_self_att=force_self_att
        )
        # q:    [bq * h, tq, d]
        # fk:   [bk * m * h, tk, d]
        # fv:   [bk * m * h, tk, d]
        # fpad: [bk, m, tk]
        assert not torch.isnan(q).any()
        assert not torch.isnan(f_key).any()
        assert not torch.isnan(f_value).any()
        if key_padding_mask is not None:
            assert not torch.isnan(key_padding_mask).any()

        # f_key = f_key.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).contiguous().permute(0, 2, 1, 3, 4)
        # f_key = f_key.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)
        #
        # f_value = f_value.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).contiguous().permute(0, 2, 3, 1, 4)
        # f_value = f_value.view(key_bsz * self.num_heads, tk * nsent, self.head_dim)
        #
        f_key = f_key.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_key = f_key.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)

        f_value = f_value.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 3, 1, 4).contiguous()
        f_value = f_value.view(key_bsz * self.num_heads, tk * nsent, self.head_dim)

        # q:    [bq * h, tq, d]
        # fk:   [bk * h, m, tk, d]
        # fv:   [bk * h, tk * m, d]
        # fpad: [bk, m, tk]

        (attn, attn_weights) = self.compute_dptree_att(
            q, f_key, f_value, flat_indices, gt_indices, attn_mask, key_padding_mask,
            src_len, tgt_len, query_bsz, need_weights)

        assert not torch.isnan(attn).any()

        return attn, attn_weights


class DPTreeSeparateOnlyKeyMatSumAttention(DPTreeSeparateOnlyKeyAttention):

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len):
        """

        :param q:           [bq * h, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, tq, d = q.size()
        q = q.unsqueeze(1)
        # q:                [bq * h, 1, tq, d]
        linear_scores = torch.matmul(q, k.transpose(2, 3))

        matrix_scores = self.__class__.scores_affinity2dptable(linear_scores, fl_idx, seq_len)
        # matrix:           [b, * h, m, tq, t, t]

        acc_fw_matrix = torch.cumsum(matrix_scores, dim=4)
        acc_bw_matrix = torch.cumsum(matrix_scores, dim=3).transpose(3, 4)
        # dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        dp_matrix = acc_fw_matrix + acc_bw_matrix
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert tq == tq_dp_mat, f'{dp_matrix.size()} ??? {q.size()} ??? {linear_scores.size()} ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, tq, t1 * t2)
        dp_scores = torch.gather(dp_linear_mat, dim=3, index=gt_idx)

        return dp_scores


class DPTreeSeparateOnlyKeyWeightSplitAttention(DPTreeSeparateOnlyKeyAttention):
    """
        use single wait for KV
        fw and bw is now different
    """
    def in_proj_k1k2v(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def prepare_dptree_qkv(self, query, key, value, key_padding_mask=None, incremental_state=None,
                           need_weights=True, static_kv=False, force_self_att=False):

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, query_bsz, embed_dim = query.size()
        ori_src_len, key_bsz, embed_dim_ = key.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, query_bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same or force_self_att:
            # self-attention
            # q, k, v = self.in_proj_qkv(query)
            k1, k2, v = self.in_proj_qkv(query)
            # q = query.contiguous()
        elif kv_same:
            # encoder-decoder attention
            # q = self.in_proj_q(query)
            # if key is None:
            #     assert value is None
            #     k = v = None
            # else:
            #     k, v = self.in_proj_kv(key)
            raise NotImplementedError(f'encoder-decoder attention not yet')
        else:
            # q = self.in_proj_q(query)
            # k = self.in_proj_k(key)
            # v = self.in_proj_v(value)
            raise NotImplementedError(f'free-style attention not yet')
        q = query * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            raise NotImplementedError

        q = q.contiguous().view(tgt_len, query_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k1 is not None:
            k1 = k1.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k2 is not None:
            k2 = k2.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if v is not None:
            v = v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key_1' in saved_state:
                prev_key_1 = saved_state['prev_key_1'].view(key_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k1 = prev_key_1
                else:
                    k1 = torch.cat((prev_key_1, k1), dim=1)
            if 'prev_key_2' in saved_state:
                prev_key_2 = saved_state['prev_key_2'].view(key_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k2 = prev_key_2
                else:
                    k2 = torch.cat((prev_key_2, k2), dim=1)

            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(key_bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key_1'] = k1.view(key_bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_2'] = k2.view(key_bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(key_bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k1.size(1)

        if key_padding_mask is not None:
            # assert key_padding_mask.size(0) == key_bsz
            assert key_padding_mask.size(-1) == src_len

        if self.add_zero_attn:
            raise NotImplementedError

        return q, k1, k2, v, key_padding_mask, saved_state, src_len, tgt_len, query_bsz

    def dptree_dot_product(self, q, k1, k2, fl_idx, gt_idx, seq_len):
        """

        :param q:           [bq * h, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, tq, d = q.size()
        q = q.unsqueeze(1)
        # q:                [bq * h, 1, tq, d]
        linear_scores_1 = torch.matmul(q, k1.transpose(2, 3))
        linear_scores_2 = torch.matmul(q, k2.transpose(2, 3))

        matrix_scores_1 = self.__class__.scores_affinity2dptable(linear_scores_1, fl_idx, seq_len)
        matrix_scores_2 = self.__class__.scores_affinity2dptable(linear_scores_2, fl_idx, seq_len)
        # matrix:           [b, * h, m, tq, t, t]

        acc_fw_matrix = torch.cumsum(matrix_scores_1, dim=4)
        acc_bw_matrix = torch.cumsum(matrix_scores_2, dim=3).transpose(3, 4)
        dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert tq == tq_dp_mat, f'{dp_matrix.size()} ??? {q.size()} ??? {linear_scores_1.size()} ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, tq, t1 * t2)
        dp_scores = torch.gather(dp_linear_mat, dim=3, index=gt_idx)

        return dp_scores

    def compute_dptree_att(self, q, k1, k2, v, fl_idx, gt_idx, attn_mask, key_padding_mask, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, tq, d]
        :param k:                   [B * h, m, tk, d]
        :param v:                   [B * h, m, tk, d] ---> [b * h, tk * m, d] ?
        :param fl_idx:              [B * h, m, tq, Tk] nor [B * h, d, Tk]
        :param gt_idx:              [B * h, m, tq, tk]
        :param attn_mask:
        :param key_padding_mask:    [B, m, tk]
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        k_size = k1.size()
        node_len = k_size[2]
        nsent = k_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        attn_weights = self.dptree_dot_product(q, k1, k2, fl_idx, gt_idx, seq_len)
        assert not torch.isnan(attn_weights).any()

        # assert list(attn_weights.size()) == [bsz * self.num_heads, nsent, tgt_len, src_len],
        if list(attn_weights.size()) != [bsz * self.num_heads, nsent, tgt_len, src_len]:
            raise ValueError(f'{attn_weights.size()} != {[bsz * self.num_heads, nsent, tgt_len, src_len]}, q={q.size()}, fl_idx={fl_idx.size()}')

        if attn_mask is not None:
            raise NotImplementedError('attn_mask for decoder not yet')

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            src_lens_denom = (~exp_pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

            attn_weights = attn_weights.float().masked_fill(
                exp_pad_mask, float('-inf')).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

        assert not torch.isnan(attn_weights).any(), f'src_lens: {src_lens_denom is None}: {src_lens_denom == 0}'

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights:     [b * h, m, tgt_len, src_len]
        attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous().view(bsz * self.num_heads, tgt_len, src_len * nsent)
        # attn_weights:     [b * h, tgt_len, src_len * m]
        assert not torch.isnan(attn_weights).any()

        attn = torch.bmm(attn_weights, v)
        assert not torch.isnan(attn).any()

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        assert not self.onnx_trace
        # if (self.onnx_trace and attn.size(1) == 1):
        #     attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        # else:
        #     attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len * nsent)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def forward(self, query, key, value, flat_indices, gt_indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, force_self_att=False):
        """Input shape: Time x Batch x Channel

        :param query:           [Tq, B, C]
        :param key:             [Tk, B, m, C]
        :param value:           [Tk, B, m, C]
        :param flat_indices:    [B * h, m, Tq, Tk]
        :param gt_indices:      [B * h, m, Tq, Tk]
        :param key_padding_mask:    [B, m, Tk]
        """

        assert flat_indices.size() == gt_indices.size(), f'{flat_indices.size()} != {gt_indices.size()}'
        tq, query_bsz, dim = query.size()
        tk, key_bsz, nsent, dim_k = key.size()
        assert query_bsz == key_bsz

        assert attn_mask is None, f'not None attn_mask (decoder self-attention) not ready'

        f_key = key.view(tk, key_bsz * nsent, dim_k)
        f_value = value.view(tk, key_bsz * nsent, dim_k)
        # f_key_pad_mask = key_padding_mask = key_padding_mask.view(key_bsz * nsent, tk)
        assert not torch.isnan(query).any()

        (q, f_key_1, f_key_2, f_value, key_padding_mask, saved_state, src_len, tgt_len, query_bsz_) = self.prepare_dptree_qkv(
            query, f_key, f_value, key_padding_mask, incremental_state, need_weights, static_kv,
            force_self_att=force_self_att
        )
        # q:    [bq * h, tq, d]
        # fk:   [bk * m * h, tk, d]
        # fv:   [bk * m * h, tk, d]
        # fpad: [bk, m, tk]
        assert not torch.isnan(q).any()
        assert not torch.isnan(f_key_1).any()
        assert not torch.isnan(f_key_2).any()
        assert not torch.isnan(f_value).any()

        if key_padding_mask is not None:
            assert not torch.isnan(key_padding_mask).any()

        # f_key = f_key.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).contiguous().permute(0, 2, 1, 3, 4)
        # f_key = f_key.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)
        #
        # f_value = f_value.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).contiguous().permute(0, 2, 3, 1, 4)
        # f_value = f_value.view(key_bsz * self.num_heads, tk * nsent, self.head_dim)

        f_key_1 = f_key_1.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_key_1 = f_key_1.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)

        f_key_2 = f_key_2.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_key_2 = f_key_2.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)

        f_value = f_value.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 3, 1, 4).contiguous()
        f_value = f_value.view(key_bsz * self.num_heads, tk * nsent, self.head_dim)

        # q:    [bq * h, tq, d]
        # fk:   [bk * h, m, tk, d]
        # fv:   [bk * h, tk * m, d]
        # fpad: [bk, m, tk]

        (attn, attn_weights) = self.compute_dptree_att(
            q, f_key_1, f_key_2, f_value, flat_indices, gt_indices, attn_mask, key_padding_mask,
            src_len, tgt_len, query_bsz, need_weights)

        assert not torch.isnan(attn).any()

        return attn, attn_weights


class DPTreeSeparateOnlyKeyWeightSplitMatSumAttention(DPTreeSeparateOnlyKeyWeightSplitAttention):
    def dptree_dot_product(self, q, k1, k2, fl_idx, gt_idx, seq_len):
        """

        :param q:           [bq * h, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, tq, d = q.size()
        q = q.unsqueeze(1)
        # q:                [bq * h, 1, tq, d]
        linear_scores_1 = torch.matmul(q, k1.transpose(2, 3))
        linear_scores_2 = torch.matmul(q, k2.transpose(2, 3))

        matrix_scores_1 = self.__class__.scores_affinity2dptable(linear_scores_1, fl_idx, seq_len)
        matrix_scores_2 = self.__class__.scores_affinity2dptable(linear_scores_2, fl_idx, seq_len)
        # matrix:           [b, * h, m, tq, t, t]

        acc_fw_matrix = torch.cumsum(matrix_scores_1, dim=4)
        acc_bw_matrix = torch.cumsum(matrix_scores_2, dim=3).transpose(3, 4)
        # dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        dp_matrix = acc_fw_matrix + acc_bw_matrix
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert tq == tq_dp_mat, f'{dp_matrix.size()} ??? {q.size()} ??? {linear_scores_1.size()} ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, tq, t1 * t2)
        dp_scores = torch.gather(dp_linear_mat, dim=3, index=gt_idx)

        return dp_scores


class DPTreeSeparateOnlyKeyRightUpAttention(DPTreeSeparateOnlyKeyAttention):

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len, k_pad_mask):
        """

        :param q:           [bq * h, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param k_pad_mask:  [B, m, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, tq, d = q.size()
        bkh, m, tk, _d = k.size()
        bsz = bqh // self.num_heads
        q = q.unsqueeze(1)
        # q:                [bq * h, 1, tq, d]
        linear_scores = torch.matmul(q, k.transpose(2, 3))
        # scores:           [bq * h, m, tq, tk]
        if k_pad_mask is not None:
            pad_mask = k_pad_mask.unsqueeze(1).unsqueeze(3)
            # [B, 1, m, 1, tk]
            linear_scores = linear_scores.view(bsz, self.num_heads, m, tq, tk)
            linear_scores = linear_scores.masked_fill(pad_mask, 0.0).type_as(linear_scores)

            linear_scores = linear_scores.view(bsz * self.num_heads, m, tq, tk)

        matrix_scores = self.__class__.scores_affinity2dptable(linear_scores, fl_idx, seq_len)
        # matrix:           [b, * h, m, tq, t, t]
        """
        a b c d
        0 f g h
        0 0 i j
        0 0 0 k

        a 0 0 0
        b f 0 0
        c g i 0
        d h j k
        """
        matrix = torch.cumsum(matrix_scores, dim=4)
        matrix = torch.flip(matrix, (3,))
        matrix = torch.cumsum(matrix, dim=3)
        matrix = torch.flip(matrix, (3,))

        # acc_fw_matrix = torch.cumsum(matrix_scores, dim=4)
        # acc_bw_matrix = torch.cumsum(matrix_scores, dim=3).transpose(3, 4)
        # dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        dp_matrix = matrix
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert tq == tq_dp_mat, f'{dp_matrix.size()} ??? {q.size()} ??? {linear_scores.size()} ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, tq, t1 * t2)
        dp_scores = torch.gather(dp_linear_mat, dim=3, index=gt_idx)

        return dp_scores

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, tq, d]
        :param k:                   [B * h, m, tk, d]
        :param v:                   [B * h, m, tk, d] ---> [b * h, tk * m, d] ?
        :param fl_idx:              [B * h, m, tq, Tk] nor [B * h, d, Tk]
        :param gt_idx:              [B * h, m, tq, tk]
        :param attn_mask:
        :param key_padding_mask:    [B, m, tk]
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        k_size = k.size()
        node_len = k_size[2]
        nsent = k_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        attn_weights = self.dptree_dot_product(q, k, fl_idx, gt_idx, seq_len, key_padding_mask)
        assert not torch.isnan(attn_weights).any()

        if list(attn_weights.size()) != [bsz * self.num_heads, nsent, tgt_len, src_len]:
            raise ValueError(f'{attn_weights.size()} != {[bsz * self.num_heads, nsent, tgt_len, src_len]}, q={q.size()}, fl_idx={fl_idx.size()}')

        if attn_mask is not None:
            raise NotImplementedError('attn_mask for decoder not yet')

        # if key_padding_mask is not None:
        #     attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)
        #
        #     exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
        #     assert not self.onnx_trace
        #
        #     # src_lens = (1.0 - exp_key_padding_mask.type_as(attn_weights)).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=1e9)
        #     src_lens = (~exp_key_padding_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=1e9)
        #     src_lens = src_lens.sqrt()
        #     attn_weights /= src_lens
        #
        #     attn_weights = attn_weights.float().masked_fill(
        #         exp_key_padding_mask, float('-inf')).type_as(attn_weights)
        #
        #     attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)
        #
        # else:
        #     attn_weights /= torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device).sqrt()
        #     src_lens = None
        # assert not torch.isnan(attn_weights).any(), f'src_lens: {src_lens is None}: {src_lens == 0}'
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            src_lens_denom = (~exp_pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

            attn_weights = attn_weights.float().masked_fill(
                exp_pad_mask, float('-inf')).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
            src_lens_denom = self.norm_src_len(src_lens_denom)
            attn_weights /= src_lens_denom

        assert not torch.isnan(attn_weights).any(), f'src_lens: {src_lens_denom is None}: {src_lens_denom == 0}'

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights:     [b * h, m, tgt_len, src_len]
        attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous().view(bsz * self.num_heads, tgt_len, src_len * nsent)
        # attn_weights:     [b * h, tgt_len, src_len * m]
        assert not torch.isnan(attn_weights).any()

        attn = torch.bmm(attn_weights, v)
        assert not torch.isnan(attn).any()

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        assert not self.onnx_trace
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len * nsent)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
            # weights:      [b, 1, s * n]
        else:
            attn_weights = None

        return attn, attn_weights
