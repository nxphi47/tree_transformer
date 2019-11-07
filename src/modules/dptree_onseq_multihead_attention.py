import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

from fairseq.modules.multihead_attention import *

DEBUG = False
from .dptree_multihead_attention import *
from .dptree_sep_multihead_attention import *
from .dptree_individual_multihead_attention import *


class DPTreeOnSeqAttention(DPTreeIndividualOnlyKeyAttention):

    def __init__(
            self, args, embed_dim, num_heads, dropout=0., bias=True,
            add_bias_kv=False, add_zero_attn=False, dptree_dim=None):
        super().__init__(args, embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn)
        self.on_seq = getattr(args, 'on_seq', 'key')
        self.divide_src_len = getattr(args, 'divide_src_len', True)

        if dptree_dim is None:
            dptree_dim = getattr(args, 'dptree_dim', embed_dim)
        self.dptree_dim = dptree_dim
        assert self.on_seq in ['query', 'key', 'value'], f'{self.on_seq}'

    @classmethod
    def indices2gt_indices(cls, indices, seq_len, query_len, heads, nsent, head_dim=None):
        """

        :param indices:     [b, m, tk, 2]
        :param seq_len:
        :param query_len:
        :param heads:
        :param head_dim:
        :return:    [B * h, m, d, Tk]
        """
        gt_idx = (indices[:, :, :, 0] * seq_len + indices[:, :, :, 1]).unsqueeze_(1).unsqueeze_(3)
        # size = gt_idx.size()
        bsz, _, nsent, _, tk = gt_idx.size()
        # gt_idx = gt_idx.expand(-1, query_len * heads, -1).contiguous().view(size[0] * heads, query_len, size[-1])
        # gt_idx = gt_idx.expand(-1, heads, nsent, query_len, -1).contiguous().view(bsz * heads, nsent, query_len, tk)
        gt_idx = gt_idx.expand(-1, heads, -1, head_dim, -1).contiguous().view(bsz * heads, nsent, head_dim, tk)
        return gt_idx

    @classmethod
    def indices2flat_indices(cls, indices, seq_len, head_dim, heads, nsent, query_len=None):
        # assert query_len is not None
        fl_idx = cls.indices2gt_indices(indices, seq_len, query_len, heads, nsent=nsent, head_dim=head_dim)
        return fl_idx

    def extra_repr(self):
        return f'on_seq={self.on_seq},dptree_dim={self.dptree_dim}'

    @classmethod
    def scores_affinity2dptable(cls, affinity, flat_indices, seq_len):
        """

        :param affinity:        [b, m, d, tk]
        :param flat_indices:    [b, m, d, tk]
        :param seq_len:
        :return:
        """
        size = affinity.size()
        bsz, aff_nsent, d, _tk = affinity.size()
        batch, nsent, d, __tk = flat_indices.size()
        assert bsz == batch, f'seq_size {size}, flidx_size {flat_indices.size()}'
        assert aff_nsent == nsent, f'{affinity.size()} ??? {flat_indices.size()}'

        zeros = torch.zeros(batch, nsent, d, seq_len * seq_len, dtype=affinity.dtype, device=affinity.device)
        # zeros:        [b, m, d, T**2]
        matrix = zeros.scatter_(dim=3, index=flat_indices, src=affinity)
        # matrix:       [b, m, d, T ** 2]
        matrix = matrix.view(batch, nsent, d, seq_len, seq_len)
        # matrix:   [B, m, tq, d, seq_len]
        return matrix

    def convert_dptree_states(self, tensor, fl_idx, gt_idx, seq_len):
        """

        :param tensor:      [b * h, m, t, d]
        :param fl_idx:      [b * h, m, d, t]
        :param gt_idx:      [b * h, m, d, t]
        :param seq_len:
        :return: states:    [b * h, m, t, d]
        """
        bh, m, t, d = tensor.size()
        tensor = tensor.transpose(2, 3)
        # tensor:           [b * h, m, d, t]
        matrix_scores = self.__class__.scores_affinity2dptable(tensor, fl_idx, seq_len)
        # matrix_scores:    [b * h, m, d, t, t]
        acc_fw_matrix = torch.cumsum(matrix_scores, dim=4)
        acc_bw_matrix = torch.cumsum(matrix_scores, dim=3).transpose(3, 4)
        dp_matrix = torch.matmul(acc_fw_matrix, acc_bw_matrix)
        bszh, nsent, tq_dp_mat, t1, t2 = dp_matrix.size()
        assert t1 == t2, f"{t1} != {t2}"
        assert d == tq_dp_mat, f'{dp_matrix.size()} ??? {tensor.size()} ???  ??? {fl_idx.size()} ??? {gt_idx.size()}'

        dp_linear_mat = dp_matrix.view(bszh, nsent, d, t1 * t2)
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

        obj = {
            'query': q,
            'key': k,
            'value': v
        }

        obj[self.on_seq] = self.convert_dptree_states(obj[self.on_seq], fl_idx, gt_idx, seq_len)

        # k: [b * h, m, d, tk]
        q = obj['query']
        k = obj['key']
        v = obj['value']

        # attn_weights = self.dptree_dot_product(q, k, fl_idx, gt_idx, seq_len)
        # print(f'q={q.size()}, k={k.size()}, v={v.size()}')
        # attn_weights = torch.matmul(q, k.transpose(2, 3))
        attn_weights = torch.matmul(q, k)

        assert not torch.isnan(attn_weights).any()

        # assert list(attn_weights.size()) == [bsz * self.num_heads, nsent, tgt_len, src_len],
        if list(attn_weights.size()) != [bsz * self.num_heads, nsent, tgt_len, src_len]:
            raise ValueError(
                f'{attn_weights.size()} != {[bsz * self.num_heads, nsent, tgt_len, src_len]},{q.size()}{fl_idx.size()}')

        if attn_mask is not None:
            raise NotImplementedError('attn_mask for decoder not yet')

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            if self.divide_src_len:
                src_lens = (~exp_key_padding_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=1e9)
                src_lens = src_lens.sqrt()
                attn_weights /= src_lens

            attn_weights = attn_weights.float().masked_fill(exp_key_padding_mask, float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            if self.divide_src_len:
                attn_weights /= torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device).sqrt()
            # assert not torch.isnan(attn_weights).any()
            src_lens = None

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            if self.divide_src_len:
                src_lens_denom = (~exp_pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
                src_lens_denom = self.norm_src_len(src_lens_denom)
                attn_weights /= src_lens_denom

            attn_weights = attn_weights.float().masked_fill(
                exp_pad_mask, float('-inf')).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            if self.divide_src_len:
                src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
                src_lens_denom = self.norm_src_len(src_lens_denom)
                attn_weights /= src_lens_denom

        assert not torch.isnan(attn_weights).any()

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
        attn = self.out_proj(attn)

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
        (f_query, f_key, f_value, key_padding_mask, saved_state, src_len, tgt_len, qbsz_) = self.prepare_dptree_qkv(
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


class DPTreeOnSeqCrossTreeAttention(DPTreeOnSeqAttention):

    def compute_dptree_att(self, q, k, v, fl_idx, gt_idx, attn_mask, key_padding_mask, src_len, tgt_len, bsz,
                           need_weights):
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
        # nsent = k_size[1]
        bh, qnsent, tq, d = q.size()
        bh_, nsent, tk, d_ = q.size()
        assert bh == bh_, f'{bh} != {bh_}'
        assert d == d_, f'{d} != {d_}'
        assert qnsent == nsent, f'{qnsent} != {nsent}'

        seq_len = int((node_len + 1) // 2) + 1

        obj = {
            'query': q,
            'key': k,
            'value': v
        }

        obj[self.on_seq] = self.convert_dptree_states(obj[self.on_seq], fl_idx, gt_idx, seq_len)

        q = obj['query'].view(bh, qnsent * tq, d)
        k = obj['key'].transpose(2, 3).view(bh, nsent * tk, d)
        v = obj['value'].view(bh, nsent * tk, d)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        # attw:     [b * h, m * tq, m * tk]

        assert not torch.isnan(attn_weights).any()

        # assert list(attn_weights.size()) == [bsz * self.num_heads, nsent, tgt_len, src_len],
        if list(attn_weights.size()) != [bsz * self.num_heads, qnsent * tgt_len, nsent * src_len]:
            raise ValueError(
                f'{attn_weights.size()} != {[bsz * self.num_heads, qnsent * tgt_len, nsent * src_len]},{q.size()}{fl_idx.size()}')

        if attn_mask is not None:
            raise NotImplementedError('attn_mask for decoder not yet')

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, qnsent * tgt_len, nsent * src_len)
            # attn_weights:     [b, h, m * tq, m * tk]
            # [B, m, tk]
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            pad_mask = pad_mask.view(bsz, 1, 1, nsent * src_len)
            # [b, 1, 1, m * tk]
            assert not self.onnx_trace

            if self.divide_src_len:
                src_lens = (~pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, zmax=1e9)
                src_lens = src_lens.sqrt()
                attn_weights /= src_lens

            attn_weights = attn_weights.float().masked_fill(pad_mask, float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, qnsent * tgt_len, nsent * src_len)

        else:
            if self.divide_src_len:
                attn_weights /= torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device).sqrt()
            # assert not torch.isnan(attn_weights).any()
            src_lens = None

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)

            exp_pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(3)
            assert not self.onnx_trace

            if self.divide_src_len:
                src_lens_denom = (~exp_pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
                src_lens_denom = self.norm_src_len(src_lens_denom)
                attn_weights /= src_lens_denom

            attn_weights = attn_weights.float().masked_fill(
                exp_pad_mask, float('-inf')).type_as(attn_weights)

            attn_weights = attn_weights.view(bsz * self.num_heads, nsent, tgt_len, src_len)

        else:
            if self.divide_src_len:
                src_lens_denom = torch.tensor(node_len, dtype=attn_weights.dtype, device=attn_weights.device)
                src_lens_denom = self.norm_src_len(src_lens_denom)
                attn_weights /= src_lens_denom

        assert not torch.isnan(attn_weights).any()


        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        assert not torch.isnan(attn_weights).any()

        attn = torch.bmm(attn_weights, v)
        assert not torch.isnan(attn).any()
        # attn:             [b * h, m * tq, d]
        attn = attn.view(bsz * self.num_heads, nsent, tgt_len, self.head_dim)

        assert list(attn.size()) == [bsz * self.num_heads, nsent, tgt_len, self.head_dim]
        assert not self.onnx_trace
        # attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = attn.permute(2, 1, 0, 3).contiguous().view(tgt_len, nsent, bsz, self.embed_dim)
        # attn:             [tq, m, b, h * dim]
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, nsent, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights