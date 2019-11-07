import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules.multihead_attention import *

DEBUG = False
from .dptree_multihead_attention import *
from .dptree_sep_multihead_attention import *


class DPTreeIndividualOnlyKeyAttention(DPTreeSeparateOnlyKeyAttention):

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
        gt_idx = gt_idx.expand(-1, heads, -1, query_len, -1).contiguous().view(bsz * heads, nsent, query_len, tk)
        return gt_idx

    @classmethod
    def indices2flat_indices(cls, indices, seq_len, head_dim, heads, nsent, query_len=None):
        assert query_len is not None
        fl_idx = cls.indices2gt_indices(indices, seq_len, query_len, heads, nsent=nsent)
        return fl_idx

    def dptree_dot_product(self, q, k, fl_idx, gt_idx, seq_len):
        """

        :param q:           [bq * h, m, tq, d]
        :param k:           [bk * h, m, tk, d]
        :param fl_idx:      [bk * h, m, tq, tk]
        :param gt_idx:      [bk * h, m, tq, tk]
        :param seq_len:
        :return:dp_scores   [bk * h, m, tq, tk]
        """
        bqh, m, tq, d = q.size()
        # q = q.unsqueeze(1)
        # q:                [bq * h, m, tq, d]
        linear_scores = torch.matmul(q, k.transpose(2, 3))
        # linear_scores:    [bq * h, m, tq, tk]

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

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, src_len, tgt_len,
                           bsz, need_weights):
        """

        :param q:                   [B * h, m, tq, d]
        :param k:                   [B * h, m, tk, d]
        :param v:                   [B * h, m, tk, d]
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

        # if key_padding_mask is not None:
        #     attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)
        #
        #     exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
        #     # exp_key_padding_mask: [bsz, 1, nsent, 1, src_len]
        #     assert not self.onnx_trace
        #
        #     # src_lens = (1.0 - exp_key_padding_mask.type_as(attn_weights)).sum(dim=-1, keepdim=True)
        #     # src_lens = (1.0 - exp_key_padding_mask.type_as(attn_weights)).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=1e9)
        #     src_lens_denom = (~exp_key_padding_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
        #     # assert not (src_lens.int() == 0).any(), f'{key_padding_mask}'
        #     src_lens_denom = src_lens_denom.sqrt_()
        #     attn_weights /= src_lens_denom
        #
        #     attn_weights = attn_weights.float().masked_fill(
        #         exp_key_padding_mask, float('-inf')).type_as(attn_weights)
        #
        #     attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)
        #
        # else:
        #     src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
        #     src_lens_denom = src_lens_denom.sqrt_()
        #     attn_weights /= src_lens_denom
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
            # src_lens = None
            # assert not torch.isnan(attn_weights).any()
        assert not torch.isnan(attn_weights).any(), f'src_lens: {src_lens_denom is None}: {src_lens_denom == 0}'

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights:     [b * h, m, tgt_len, src_len]
        # values:           [b * h, m, tk, d]

        # attn_weights = attn_weights.permute(0, 2, 3, 1).contiguous().view(bsz * self.num_heads, tgt_len, src_len * nsent)
        # attn_weights:     [b * h, tgt_len, src_len * m]
        assert not torch.isnan(attn_weights).any()

        attn = torch.matmul(attn_weights, v)
        assert not torch.isnan(attn).any()
        # attn:             [b * h, m, tq, d]

        assert list(attn.size()) == [bsz * self.num_heads, nsent, tgt_len, self.head_dim]
        assert not self.onnx_trace
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = attn.permute(2, 1, 0, 3).contiguous().view(tgt_len, nsent, bsz, self.embed_dim)
        # attn:             [tq, m, b, h * dim]

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def forward(
            self, query, key, value, flat_indices, gt_indices, key_padding_mask=None,
            incremental_state=None,
            need_weights=True, static_kv=False, attn_mask=None, force_self_att=False):
        """Input shape: Time x Batch x Channel

        :param query:           [Tq, B, m, C]
        :param key:             [Tk, B, m, C]
        :param value:           [Tk, B, m, C]
        :param flat_indices:    [B * h, m, Tq, Tk]
        :param gt_indices:      [B * h, m, Tq, Tk]
        :param key_padding_mask:    [B, m, Tk]
        """

        assert flat_indices.size() == gt_indices.size(), f'{flat_indices.size()} != {gt_indices.size()}'
        tq, query_bsz, qnsent, dim = query.size()
        tk, key_bsz, nsent, dim_k = key.size()
        assert query_bsz == key_bsz
        assert qnsent == nsent

        assert attn_mask is None, f'not None attn_mask (decoder self-attention) not ready'

        f_query = query.view(tq, query_bsz * qnsent, dim)
        f_key = key.view(tk, key_bsz * nsent, dim_k)
        f_value = value.view(tk, key_bsz * nsent, dim_k)

        # f_key_pad_mask = key_padding_mask = key_padding_mask.view(key_bsz * nsent, tk)
        assert not torch.isnan(query).any()
        (
        f_query, f_key, f_value, key_padding_mask, saved_state, src_len, tgt_len, query_bsz_) = self.prepare_dptree_qkv(
            f_query, f_key, f_value, key_padding_mask, incremental_state, need_weights, static_kv,
            force_self_att=force_self_att
        )
        # q:    [bq * m * h, tq, d]
        # fk:   [bk * m * h, tk, d]
        # fv:   [bk * m * h, tk, d]
        # fpad: [bk, m, tk]
        assert not torch.isnan(f_query).any()
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

        f_query = f_query.view(query_bsz, qnsent, self.num_heads, tq, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        # f_query:          [b, h, m, tq, d]
        f_query = f_query.view(query_bsz * self.num_heads, qnsent, tq, self.head_dim)

        f_key = f_key.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        # f_key:            [b, h, m, tk, d]
        f_key = f_key.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)

        f_value = f_value.view(key_bsz, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        # f_value:            [b, h, m, tk, d]
        f_value = f_value.view(key_bsz * self.num_heads, nsent, tk, self.head_dim)

        # q:    [bq * h, tq, d]
        # fk:   [bk * h, m, tk, d]
        # fv:   [bk * h, m, tk, d]
        # fpad: [bk, m, tk]

        (attn, attn_weights) = self.compute_dptree_att(
            f_query, f_key, f_value, flat_indices, gt_indices, attn_mask, key_padding_mask,
            src_len, tgt_len, query_bsz, need_weights)

        assert not torch.isnan(attn).any()
        # attn:             [tq, m, b, h * dim]
        return attn, attn_weights


class DPTreeIndividualRNNOnlyKeyAttention(DPTreeIndividualOnlyKeyAttention):

    def forward(self, query, key, value, flat_indices, gt_indices, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, force_self_att=False):
        """Input shape: Time x Batch x Channel
            *** Only Apply to Encoder-Self-Attention
        :param query:           [Tq, B, m, C]
        :param key:             [Tk, B, m, C]
        :param value:           [Tk, B, m, C]
        :param flat_indices:    [B * h, m, Tq, Tk]
        :param gt_indices:      [B * h, m, Tq, Tk]
        :param key_padding_mask:    [B, m, Tk]
        """

        assert flat_indices.size() == gt_indices.size(), f'{flat_indices.size()} != {gt_indices.size()}'
        tq, query_bsz, qnsent, dim = query.size()
        tk, key_bsz, nsent, dim_k = key.size()
        assert query_bsz == key_bsz
        assert qnsent == nsent

        assert attn_mask is None, f'not None attn_mask (decoder self-attention) not ready'
        assert incremental_state is None, f'not None incremental_state'

        queries = query.chunk(qnsent, 2)
        keys = key.chunk(nsent, 2)
        values = value.chunk(nsent, 2)
        fl_indices_list = flat_indices.chunk(nsent, 1)
        gt_indices_list = gt_indices.chunk(nsent, 1)
        pad_masks = key_padding_mask.chunk(nsent, 1)

        assert attn_mask is None

        reduce_lengths = []
        original_lengths = []
        attentions = []
        for i, (q, k, v, fi, gt, mask) in enumerate(zip(
                queries, keys, values, fl_indices_list, gt_indices_list, pad_masks
        )):
            # reduce padding....
            # mask:     [b, 1, tk]
            ori_length = mask.size(-1)
            original_lengths.append(ori_length)
            length_mask = (~mask).int().sum(-1) + 1
            max_length = length_mask.max()
            reduce_lengths.append(max_length)

            q = q[:max_length]
            k = k[:max_length]
            v = v[:max_length]
            # FIXME: check this one
            fi = fi[:, :, :max_length, :max_length]
            gt = gt[:, :, :max_length, :max_length]
            mask = mask[:, :, :max_length]

            attn, attn_weights_ = super().forward(
                q, k, v, fi, gt, mask, incremental_state,
                need_weights, static_kv, attn_mask, force_self_att)

            # pad_attention
            # attn:             [tq, m, b, h * dim]
            attn = F.pad(attn, [0, 0, 0, 0, 0, 0, 0, tq - max_length])
            attentions.append(attn)
        # attn:             [tq, m, b, h * dim]

        out_attn = torch.cat(attentions, 1)
        return out_attn, None




