import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules.multihead_attention import *

DEBUG = False

def maybe_print(s):
    if DEBUG:
        print('dptree_multihead_attention.py::' + s)


def default_triangular_fn(
        q, length, **kwargs
):
    """

    :param q: 		[b*h, lq, d]
    :param length:
    :return: triu:	[b*h, lq, d, l, l]
    """
    # mask = tf.sequence_mask(tf.range(length, 0, -1, dtype=tf.int32), length, dtype=tf.float32)
    # triu = tf.reverse(mask, [1])
    # triu = triu[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, ...]
    # feed_q = tf.transpose(q, [0, 1, 3, 2])
    # q_triu = feed_q[..., tf.newaxis, tf.newaxis] * triu
    triu = torch.triu(torch.ones(length, length)).cuda().view(1, 1, 1, length, length)
    ex_q = q.view(*(list(q.size()) + [1, 1]))
    q_triu = ex_q * triu
    return q_triu


class DPTreeMultiheadAttention(nn.Module):
    def __init__(self, args, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()

        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
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
        # raise NotImplementedError

    def prepare_qkv_legacy(self, query, key, value, indices, key_padding_mask=None, incremental_state=None,
                           need_weights=True, static_kv=False, attn_mask=None):
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
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
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        return q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz

    def compute_att_legacy(self, q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz,
                           need_weights):
        """

        :param q:                   [B*h, Tq, C]
        :param k:                   [B*h, tk, C]
        :param v:                   [B*h, tk, C]
        :param attn_mask:
        :param key_padding_mask:
        :param saved_state:
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def query2matrix(self, q, length):
        q_triu = default_triangular_fn(q, length)
        return q_triu

    @classmethod
    def seq2matrix(cls, seq, flat_indices, seq_len):
        """
            scatter_ is not scatter_nd!
            How to...
            1.
                zeros   -> [B, C, T**2]
                fl_idx  -> [B, Tk, 1] , [0] * seq_len + [1]
                seq     -> [B, Tk, C].transpose(-1, -2)
                seq     -> [B, C, Tk]

                need
                fl_idx  -> [B, C, Tk]
                zero[i][j][fl_idx[j,j,k]] = src[j,j,k]

                zeros.scatter_(dim=-1, index=fl_idx, src=seq)

        :param seq:             [B, Tk, C]
        :param flat_indices:    [B, C, Tk]
        :param seq_len: int
        :return:                [B, C, T, T]
        """

        # FIXME: seq_len = tf.to_int32((node_len + 1) // 2) + 1
        size = seq.size()
        fl_idx_size = flat_indices.size()
        batch = size[0]
        node_len = size[1]
        hidden_dim = size[2]
        assert size[0] == fl_idx_size[0], f'seq_size {size}, flidx_size {fl_idx_size}'

        zeros = torch.zeros(batch, hidden_dim, seq_len * seq_len).cuda()
        # zeros:    [B, C, T ** 2]
        seq_t = seq.transpose(1, 2)
        matrix = zeros.scatter_(dim=2, index=flat_indices, src=seq_t)
        # matrix:   [B, C, T]
        matrix = matrix.view(batch, hidden_dim, seq_len, seq_len)
        # matrix:   [B, C, seq_len, seq_len]

        return matrix

    @classmethod
    def indices2flat_indices(cls, indices, seq_len, head_dim, heads, **kwargs):
        """

        :param indices:     [B, Tk, 2]
        :param seq_len:     batched seq_len
        :param head_dim:  C
        :param heads        h
        :return:            [B * h, d, Tk]
        """
        fl_idx = (indices[:, :, 0] * seq_len + indices[:, :, 1]).unsqueeze_(1)
        # fl_idx = [B, 1, Tk]
        size = fl_idx.size()
        hidden_dim = head_dim * heads
        fl_idx = fl_idx.expand(-1, hidden_dim, -1).contiguous().view(size[0] * heads, head_dim, size[-1])
        # fl_idx = fl_idx.expand(-1, hidden_dim, -1).view(size[0] * heads, head_dim, size[-1])
        return fl_idx

    @classmethod
    def indices2gt_indices(cls, indices, seq_len, query_len, heads, **kwargs):
        gt_idx = (indices[:, :, 0] * seq_len + indices[:, :, 1]).unsqueeze_(1)
        size = gt_idx.size()
        gt_idx = gt_idx.expand(-1, query_len * heads, -1).contiguous().view(size[0] * heads, query_len, size[-1])
        # gt_idx = gt_idx.expand(-1, query_len * heads, -1).view(size[0] * heads, query_len, size[-1])
        return gt_idx

    def prepare_dptree_qkv(self, query, key, value, flat_indices, key_padding_mask=None, incremental_state=None,
                           need_weights=True, static_kv=False, attn_mask=None):

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
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
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        # q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # if k is not None:
        #     k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # if v is not None:
        #     v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # C = h * d
        # q: [B * h, tq, d]
        # k: [B * h, tk, d]
        # v: [B * h, tk, d]

        # now q: [T, b, h*d]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # C = h * d
        # q: [B * h, tq, d]
        # k: [B * h, tk, d]
        # v: [B * h, tk, d]
        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            # default False
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        return q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len):
        """

        :param q:           [B * h, tq, d]
        :param k:           [B * h, tk, d]
        :param fl_idx:      [B * h, d, tk]
        :param gt_idx:      [B * h, tq, tk]
        :param seq_len:
        :return:            [B * h, lq, tk]
        """
        maybe_print(f'q-{q.size()}, k-{k.size()}, fl_idx-{fl_idx.size()}, gt_idx-{gt_idx.size()}, seq_len={seq_len}')
        k_matrix = DPTreeMultiheadAttention.seq2matrix(k, fl_idx, seq_len)
        # k_matrix  [B * h, d, T, T]
        maybe_print(f'k_matrix = {k_matrix.size()}, seq_len = {seq_len}')
        k_matrix = k_matrix.unsqueeze_(1)
        # k_matrix  [B * h, 1,  d, T, T]
        q_matrix = self.query2matrix(q, seq_len)
        # q_matrix  [B * h, lq, d, l, l]
        maybe_print(f'q_matrix = {q_matrix.size()}, seq_len = {seq_len}')

        mat_aq = torch.matmul(k_matrix, q_matrix)
        # mat_aq =  [B * h, lq, d, t, t]
        mat_aqa = torch.matmul(mat_aq, q_matrix)
        maybe_print(f'mat_aqa = {mat_aqa.size()}')

        mat_aqa_s = mat_aqa.sum(dim=-3)
        mat_aqa_s_size = mat_aqa_s.size()
        # mat_aqa_s [B * h, lq, t, t]
        maybe_print(f'mat_aqa_s_size = {mat_aqa_s_size}')
        mat_aqa_s_fl = mat_aqa_s.view(mat_aqa_s_size[0], mat_aqa_s_size[1], mat_aqa_s_size[2] ** 2)
        # mat_aqa_s_fl [B * h, lq, t*t]
        maybe_print(f'mat_aqa_s_fl {mat_aqa_s_fl.size()}, gt_idx {gt_idx.size()}')
        gt_w = torch.gather(input=mat_aqa_s_fl, dim=2, index=gt_idx)
        return gt_w

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, saved_state, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, tq, d]
        :param k:                   [B * h, tk, d]
        :param v:                   [B * h, tk, d]
        :param fl_idx:              [B * h, d, Tk]
        :param gt_idx:              [B * h, tq, tk]
        :param attn_mask:
        :param key_padding_mask:
        :param saved_state:
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        k_size = k.size()
        node_len = k_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        attn_weights = self.dptree_dot_product(q, k, fl_idx, gt_idx, seq_len)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def forward(self, query, key, value, flat_indices, gt_indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.

        :param query:           [Tq, B, C]
        :param key:             [Tk, B, C]
        :param value:           [Tk, B, C]
        :param flat_indices:    [B, C, Tk] - NOT [Tk, B, 2]
        :param gt_indices:      [B, Tq, Tk]- NOT [Tk, B, 2]
        :param key_padding_mask:    [B, Tk]
        """
        # TODO: to do in multihead_attention.py

        (q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz) = self.prepare_dptree_qkv(
            query, key, value, flat_indices, key_padding_mask, incremental_state, need_weights, static_kv, attn_mask
        )

        (attn, attn_weights) = self.compute_dptree_att(
            q, k, v, flat_indices, gt_indices, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz,
            need_weights)

        return attn, attn_weights

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


class DPTreeOnlyKeyAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
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

    def prepare_dptree_qkv(self, query, key, value, flat_indices, key_padding_mask=None, incremental_state=None,
                           need_weights=True, static_kv=False, attn_mask=None):

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
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
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            raise NotImplementedError

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            raise NotImplementedError

        return q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz

    @classmethod
    def seq2matrix(cls, seq, flat_indices, seq_len):
        """
            scatter_ is not scatter_nd!
            How to...
            1.
                zeros   -> [B, C, T**2]
                fl_idx  -> [B, Tk, 1] , [0] * seq_len + [1]
                seq     -> [B, Tk, C].transpose(-1, -2)
                seq     -> [B, C, Tk]

                need
                fl_idx  -> [B, C, Tk]
                zero[i][j][fl_idx[j,j,k]] = src[j,j,k]

                zeros.scatter_(dim=-1, index=fl_idx, src=seq)

        :param seq:             [B, Tk, C]
        :param flat_indices:    [B, C, Tk]
        :param seq_len: int
        :return:                [B, C, T, T]
        """

        # FIXME: seq_len = tf.to_int32((node_len + 1) // 2) + 1
        size = seq.size()
        fl_idx_size = flat_indices.size()
        batch = size[0]
        node_len = size[1]
        hidden_dim = size[2]
        assert size[0] == fl_idx_size[0], f'seq_size {size}, flidx_size {fl_idx_size}'

        zeros = torch.zeros(batch, hidden_dim, seq_len * seq_len, dtype=seq.dtype, device=seq.device)
        # zeros:    [B, C, T ** 2]
        seq_t = seq.transpose(1, 2)
        matrix = zeros.scatter_(dim=2, index=flat_indices, src=seq_t)
        # matrix:   [B, C, T]
        matrix = matrix.view(batch, hidden_dim, seq_len, seq_len)
        # matrix:   [B, C, seq_len, seq_len]

        return matrix

    @classmethod
    def indices2gt_indices(cls, indices, seq_len, query_len, heads, head_dim=None):
        gt_idx = (indices[:, :, 0] * seq_len + indices[:, :, 1]).unsqueeze_(1)
        size = gt_idx.size()
        # [b, 1, tk]
        gt_idx = gt_idx.expand(-1, query_len * heads, -1).contiguous().view(size[0] * heads, query_len, size[-1])
        # gt_idx = gt_idx.expand(-1, query_len * heads, -1).view(size[0] * heads, query_len, size[-1])
        return gt_idx

    @classmethod
    def indices2flat_indices(cls, indices, seq_len, head_dim, heads, query_len=None):
        # fl_idx = (indices[:, :, 0] * seq_len + indices[:, :, 1]).unsqueeze_(1)
        # # fl_idx = [B, 1, Tk]
        # size = fl_idx.size()
        # hidden_dim = head_dim * heads
        # fl_idx = fl_idx.expand(-1, hidden_dim, -1).contiguous().view(size[0] * heads, head_dim, size[-1])
        # fl_idx = fl_idx.expand(-1, hidden_dim, -1).view(size[0] * heads, head_dim, size[-1])
        assert query_len is not None
        fl_idx = cls.indices2gt_indices(indices, seq_len, query_len, heads)
        return fl_idx

    @classmethod
    def scores_affinity2dptable(cls, affinity, flat_indices, seq_len):
        """

        :param affinity:        [b, tq, tk]
        :param flat_indices:    [b, tq, tk]
        :param seq_len:
        :return:
        """
        size = affinity.size()
        # fl_idx_size = flat_indices.size()
        batch, node_len, hidden_dim = flat_indices.size()
        # batch = size[0]
        # node_len = size[1]
        # hidden_dim = size[2]
        assert size[0] == batch, f'seq_size {size}, flidx_size {flat_indices.size()}'

        # zeros = torch.zeros(batch, hidden_dim, seq_len * seq_len, dtype=affinity.dtype, device=affinity.device)
        zeros = torch.zeros(batch, node_len, seq_len * seq_len, dtype=affinity.dtype, device=affinity.device)
        # zeros:    [B, tq, T ** 2]
        # seq_t = affinity.transpose(1, 2)
        matrix = zeros.scatter_(dim=2, index=flat_indices, src=affinity)
        # matrix:   [B, tq, T ** 2]
        matrix = matrix.view(batch, node_len, seq_len, seq_len)
        # matrix:   [B, tq, seq_len, seq_len]
        return matrix

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len):
        """

        :param q:           [B * h, tq, d]
        :param k:           [B * h, tk, d]
        :param fl_idx:      [B * h, tq, tk] nor [B * h, d, Tk]
        :param gt_idx:      [B * h, tq, tk]
        :param seq_len:
        :return:            [B * h, lq, tk]
        """

        linear_scores = torch.bmm(q, k.transpose(1, 2))
        # linear_scores:            [B*h, tq, tk]
        matrix_scores = self.__class__.scores_affinity2dptable(linear_scores, fl_idx, seq_len)
        # matrix_scores:            [B*H, tq, t, t]

        acc_fw_matrix = torch.cumsum(matrix_scores, dim=3)
        acc_bw_matrix = torch.cumsum(matrix_scores, dim=2).transpose(2, 3)

        dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        bszh, tq, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        # dp_matrix:                [B*h, tq, t, t]
        dp_linear_mat = dp_matrix.view(bszh, tq, t1 * t2)
        # print(f'linear_scores: {linear_scores.size()}, matrix_scores: {matrix_scores.size()}, dp_matrix: {dp_matrix.size()}, dp_linear_mat: {dp_linear_mat.size()}, fl_idx: {fl_idx.size()}, gt_idx: {gt_idx.size()}')
        dp_scores = torch.gather(dp_linear_mat, dim=2, index=gt_idx)
        return dp_scores

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, saved_state, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, tq, d]
        :param k:                   [B * h, tk, d]
        :param v:                   [B * h, tk, d]
        :param fl_idx:              [B * h, tq, Tk] nor [B * h, d, Tk]
        :param gt_idx:              [B * h, tq, tk]
        :param attn_mask:
        :param key_padding_mask:
        :param saved_state:
        :param src_len:
        :param tgt_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        k_size = k.size()
        node_len = k_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        attn_weights = self.dptree_dot_product(q, k, fl_idx, gt_idx, seq_len)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # exp_key_padding_mask: [bsz, 1, 1, src_len]
            if self.onnx_trace:
                attn_weights = torch.where(
                    exp_key_padding_mask,
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(exp_key_padding_mask, float('-inf')).type_as(attn_weights)  # FP16 support: cast to float and back

            src_lens = (1.0 - exp_key_padding_mask.type_as(attn_weights)).sum(dim=-1, keepdim=True)
            src_lens = src_lens.sqrt()
            attn_weights /= src_lens
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights /= torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device).sqrt()

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, self.embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def forward(self, query, key, value, flat_indices, gt_indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.

        :param query:           [Tq, B, C]
        :param key:             [Tk, B, C]
        :param value:           [Tk, B, C]
        :param flat_indices:    [B, Tq, Tk] - NOT [Tk, B, 2]
        :param gt_indices:      [B, Tq, Tk]- NOT [Tk, B, 2]
        :param key_padding_mask:    [B, Tk]
        """

        assert flat_indices.size() == gt_indices.size(), f'{flat_indices.size()} != {gt_indices.size()}'
        # print(f'query: {query.size()}, key: {key.size()}, flat_indices: {flat_indices.size()}')

        """
        query: torch.Size([32, 8, 512]), key: torch.Size([68, 8, 512]), flat_indices: torch.Size([64, 32, 68])
        dp_matrix: torch.Size([64, 68, 35, 35]), dp_linear_mat: torch.Size([64, 68, 1225]), 
        fl_idx: torch.Size([64, 32, 68]), gt_idx: torch.Size([64, 32, 68])
        """

        (q, k, v, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz) = self.prepare_dptree_qkv(
            query, key, value, flat_indices, key_padding_mask, incremental_state, need_weights, static_kv, attn_mask
        )

        (attn, attn_weights) = self.compute_dptree_att(
            q, k, v, flat_indices, gt_indices, attn_mask, key_padding_mask, saved_state, src_len, tgt_len, bsz,
            need_weights)


        return attn, attn_weights



