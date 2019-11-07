import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils

import functools
from fairseq.models import transformer
from fairseq.modules.multihead_attention import *

DEBUG = False


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:

        m = transformer.LearnedPositionalEmbedding(
            num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        raise NotImplementedError
    else:
        m = transformer.SinusoidalPositionalEmbedding(
            embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m


def NstackLinear(in_features, out_features, bias=False):
    assert in_features == out_features
    m = nn.Linear(in_features, out_features, bias)
    with torch.no_grad():
        nn.init.xavier_uniform_(m.weight)
        # m.weight += torch.eye(in_features, dtype=m.weight.dtype, device=m.weight.device)
        m.weight.add_(torch.eye(in_features, dtype=m.weight.dtype, device=m.weight.device))
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


class WeightMask(object):
    DEFAULT = 'default'
    ALL_ALL = 'all_all'
    LEAVES_SUBTREE = 'leaves_subtree'
    LEAVES_SUBTREELEAVES = 'leaves_subtreeleaves'
    LEAVESANCESTORS_SUBTREELEAVES = 'leavesancestors_subtreeleaves'
    MUTUALANCESTORS_SUBTREE = "mutualancestors_subtree"
    ALL_SUBTREE = 'all_subtree'
    LEAVES_ALL = 'leaves_all'

    @classmethod
    def acquire_wmask_function(cls, fname, mutual_level=5):
        # fname = getattr(args, 'nstack_mask_fn', cls.DEFAULT)
        if fname == cls.DEFAULT:
            return cls.wmask_default
        elif fname == cls.ALL_ALL:
            return cls.wmask_all_all
        elif fname == cls.LEAVES_SUBTREE:
            return cls.wmask_leaves_subtree
        elif fname == cls.LEAVES_SUBTREELEAVES:
            return cls.wmask_leaves_subtreeleaves
        elif fname == cls.LEAVESANCESTORS_SUBTREELEAVES:
            return cls.wmask_leavesancestors_subtreeleaves
        elif fname == cls.MUTUALANCESTORS_SUBTREE:
            return functools.partial(cls.wmask_mutualancestors_subtree, mutual_level=mutual_level)
        elif fname == cls.ALL_SUBTREE:
            return cls.wmask_all_subtree
        elif fname == cls.LEAVES_ALL:
            return cls.wmask_leaves_all
        else:
            raise ValueError(f'{fname} invalid')

    @classmethod
    def wmask_default(cls, self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        att_nelems = nsent * (tk + nk)
        # assert tq == att_nelems, f'{tq} != {att_nelems} should be self-attention'
        if key_pad is not None:
            assert node_pad is not None
            attn_weights = attn_weights.view(bsz, self.num_heads, tq, att_nelems)
            pad_mask = torch.cat([key_pad, node_pad], 2).view(bsz, 1, 1, att_nelems)
            assert not self.onnx_trace
            src_lens_denom = (~pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
            attn_weights = self.maybe_norm_src_len(attn_weights, src_lens_denom)
            attn_weights = attn_weights.float().masked_fill(pad_mask, float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, tq, att_nelems)
        else:
            src_lens_denom = torch.tensor(att_nelems, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = self.maybe_norm_src_len(attn_weights, src_lens_denom)

        return attn_weights

    @classmethod
    def wmask_all_all(cls, self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        return cls.wmask_default(self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs)

    @classmethod
    def wmask_leaves_subtree(cls, self, att_w, key_pad, node_pad, node_idx, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)
        leave_w, node_w = cls.split_attn_weights(self, att_w, bsz, tq, tk, nk, nsent)
        leave_w = cls.mask_leaves_only(self, leave_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        node_w = cls.mask_subtree_only(self, node_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        att_w = cls.merge_attn_weights(self, leave_w, node_w, bsz, tq, tk, nk, nsent)
        return att_w

    @classmethod
    def wmask_leaves_subtreeleaves(cls, self, att_w, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)
        leave_w, node_w = cls.split_attn_weights(self, att_w, bsz, tq, tk, nk, nsent)
        leave_w = cls.mask_leaves_only(self, leave_w, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        node_w = cls.mask_subtree_n_leaves(self, node_w, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        att_w = cls.merge_attn_weights(self, leave_w, node_w, bsz, tq, tk, nk, nsent)
        return att_w

    @classmethod
    def wmask_leavesancestors_subtreeleaves(cls, self, att_w, key_pad, node_pad, node_idx, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)
        leaves_weights, nodes_weights = cls.split_attn_weights(self, att_w, bsz, tq, tk, nk, nsent)
        leaves_weights = cls.mask_leaves_ancestors(self, leaves_weights, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        nodes_weights = cls.mask_subtree_n_leaves(self, nodes_weights, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        att_w = cls.merge_attn_weights(self, leaves_weights, nodes_weights, bsz, tq, tk, nk, nsent)
        return att_w

    @classmethod
    def wmask_mutualancestors_subtree(cls, self, att_w, key_pad, node_pad, node_idx, bsz, tq, tk, nk, nsent, **kwargs):
        assert 'mutual_level' in kwargs
        mutu_level = kwargs['mutual_level']
        leave_w, node_w = cls.split_attn_weights(self, att_w, bsz, tq, tk, nk, nsent)
        node_w = cls.mask_subtree_only(self, node_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        leave_w = cls.mask_mutual_ancestors(mutu_level, self, leave_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        att_w = cls.merge_attn_weights(self, leave_w, node_w, bsz, tq, tk, nk, nsent)
        return att_w

    @classmethod
    def wmask_all_subtree(cls, self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)
        leaves_weights, nodes_weights = cls.split_attn_weights(self, attn_weights, bsz, tq, tk, nk, nsent)
        leaves_weights = cls.mask_all(self, leaves_weights, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        nodes_weights = cls.mask_subtree_only(self, nodes_weights, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        attn_weights = cls.merge_attn_weights(self, leaves_weights, nodes_weights, bsz, tq, tk, nk, nsent)
        return attn_weights

    @classmethod
    def wmask_leaves_all(cls, self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)
        leaves_weights, nodes_weights = cls.split_attn_weights(self, attn_weights, bsz, tq, tk, nk, nsent)
        leaves_weights = cls.mask_leaves_only(self, leaves_weights, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        nodes_weights = cls.mask_all(self, nodes_weights, key_pad, node_pad, node_indices, bsz, tk, nk, nsent)
        attn_weights = cls.merge_attn_weights(self, leaves_weights, nodes_weights, bsz, tq, tk, nk, nsent)
        return attn_weights

    @classmethod
    def split_attn_weights(cls, self, attn_weights, bsz, tq, tk, nk, nsent):
        # att_weights:          [b * h, tq, m * (tk + nk)]
        all_weights = attn_weights.view(bsz, self.num_heads, tk + nk, nsent, nsent, tk + nk)
        leaves_weights = all_weights[:, :, :tk]
        nodes_weights = all_weights[:, :, tk:]
        return leaves_weights, nodes_weights

    @classmethod
    def merge_attn_weights(cls, self, leaves_weights, nodes_weights, bsz, tq, tk, nk, nsent):
        all_weights = torch.cat([leaves_weights, nodes_weights], 2)
        all_weights = all_weights.view(bsz * self.num_heads, (tk + nk) * nsent, nsent * (tk + nk))
        return all_weights

    @classmethod
    def mask_all(cls, self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        # weights:                  [b, h, t, m, m, tk + nk]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        if key_pad is not None:
            assert node_pad is not None
            pad_mask = torch.cat([key_pad, node_pad], 2).view(bsz, 1, 1, 1, m, tk + nk)
            src_lens_denom = (~pad_mask).type_as(weights).sum(dim=[-1, -2], keepdim=True).clamp_(min=1.0, max=10000)
            weights = self.maybe_norm_src_len(weights, src_lens_denom)
            weights = weights.float().masked_fill(pad_mask, float('-inf')).type_as(weights)
        else:
            att_nelems = m * (tk + nk)
            src_lens_denom = torch.tensor(att_nelems, dtype=weights.dtype, device=weights.device)
            weights = self.maybe_norm_src_len(weights, src_lens_denom)
        return weights

    @classmethod
    def mask_leaves_only(cls, self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        # weights:                  [b, h, t, m, m, tk + nk]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        if node_pad is not None:
            node_pad = torch.ones_like(node_pad, dtype=node_pad.dtype, device=node_pad.device)
        else:
            assert key_pad is None
            node_pad = torch.ones(bsz, m, nk, dtype=torch.uint8, device=weights.device)
            key_pad = torch.zeros(bsz, m, tk, dtype=torch.uint8, device=weights.device)
        weights = WeightMask.mask_all(self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs)
        return weights

    @classmethod
    def build_ancestors_node_pad(cls, num_heads, weights, key_pad, node_pad, node_idx, bsz, tq, tk, nk, m, **kwargs):
        # weights:                  [b, h, t, m, m, tk + nk]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        with torch.no_grad():
            device = node_idx.device
            # key_pad = key_pad if key_pad is not None else torch.zeros(bsz, m, tk, device=device).byte()
            node_pad = node_pad if node_pad is not None else torch.zeros(bsz, m, nk, device=device).byte()
            assert tq == tk, f'{tq} != {tk}.. only apply to leaves'
            # key_pad = key_pad.view(bsz, 1, 1, 1, m, tk)
            node_pad = node_pad.view(bsz, 1, 1, 1, m, nk)
            indices = node_idx.contiguous().view(bsz, num_heads, m, nk, 1, 1, 2).transpose(2, 3)

            # todo: leave_mask:          [b, h, tk, m, m, nk]
            leave_range = torch.arange(0, tk, dtype=indices.dtype, device=indices.device)
            node_npad_rg = leave_range.view(1, 1, 1, 1, tk)
            fnode_idx = node_idx.view(bsz, num_heads, m, nk, 2)
            # node_npad_rg:              [1, 1, 1,  1, tk]
            # fnode_idx:                 [b, h, m, nk,  2]
            # node_leaves:               [b, h, m, nk, tk]
            # out:                       [b, h, tk, m, 1, nk]
            node_leaves = (node_npad_rg >= fnode_idx[:, :, :, :, :1]) ^ (node_npad_rg > fnode_idx[:, :, :, :, 1:])
            node_leaves = node_leaves.view(bsz, num_heads, m, 1, nk, tk).permute(0, 1, 5, 2, 3, 4)
            node_mask = (~node_leaves) | node_pad

        return node_mask

    @classmethod
    def mask_leaves_ancestors(cls, self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        b, h, t, m__, m_, tknk = weights.size()
        # bsz, self.num_heads, tk + nk, nsent, nsent, tk + nk)
        node_pad = cls.build_ancestors_node_pad(
            self.num_heads, weights, key_pad, node_pad, node_indices, bsz, t, tk, nk, m, **kwargs)
        key_pad = key_pad if key_pad is not None else torch.zeros(bsz, m, tk, dtype=torch.uint8, device=weights.device)
        # torch.Size([32, 10, 66, 1, 1, 131]),
        # torch.Size([32, 1, 66]),
        # torch.Size([32, 10, 66, 1, 1, 65])
        # weights = cls.mask_all(self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs)
        _b, _h, _t, _m, __m, _nk = node_pad.size()
        key_pad = key_pad.view(bsz, 1, 1, 1, m, tk).expand(_b, _h, _t, _m, m, tk)

        # torch.Size([32, 10, 66, 1, 1, 131]),
        # torch.Size([32, 10, 66, 1, 1, 66]),
        # torch.Size([32, 10, 66, 1, 1, 65])

        pad_mask = torch.cat([key_pad, node_pad], 5)
        src_lens_denom = (~pad_mask).type_as(weights).sum(dim=[-1, -2], keepdim=True).clamp_(min=1.0, max=10000)
        weights = self.maybe_norm_src_len(weights, src_lens_denom)
        weights = weights.float().masked_fill(pad_mask, float('-inf')).type_as(weights)

        return weights

    @classmethod
    def build_subtree_mask(cls, num_heads, weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, m, **kwargs):
        with torch.no_grad():
            device = node_indices.device
            key_pad = key_pad if key_pad is not None else torch.zeros(bsz, m, tk, device=device).byte()
            node_pad = node_pad if node_pad is not None else torch.zeros(bsz, m, nk, device=device).byte()
            # b, h, t, m__, m_, tknk = weights.size()
            assert tq == nk, f'{tq} != {nk}.. only apply to nodes'
            key_pad = key_pad.view(bsz, 1, 1, 1, m, tk)
            node_pad = node_pad.view(bsz, 1, 1, 1, m, nk)
            indices = node_indices.contiguous().view(bsz, num_heads, m, nk, 1, 1, 2).transpose(2, 3)
            # indices:                  [b, h, t, m, 1, 1, 2]
            # todo: do leave_mask
            leave_range = torch.arange(0, tk, dtype=indices.dtype, device=indices.device)

            leave_npad_rg = leave_range.view(1, 1, 1, 1, 1, tk)
            leave_npad = (leave_npad_rg < indices[:, :, :, :, :, :, 0]) | (leave_npad_rg > indices[:, :, :, :, :, :, 1])
            leave_npad = leave_npad | node_pad.contiguous().view(bsz, 1, m, nk, 1, 1).transpose(2, 3)
            # leave_pad:                [b, h, t, m, 1, tk]
            # key_pad   :               [b, 1, 1, 1, m, tk]
            # node_pad_ :               [b, 1, t, m, 1,  1]
            leave_mask = leave_npad | key_pad
            # leave_mask:               [b, h, t, m, m, tk]
            # todo: node_mask:          [b, h, nk, m, m, nk]
            node_npad_rg = leave_range.view(1, 1, 1, 1, tk)
            fnode_idx = node_indices.view(bsz, num_heads, m, nk, 2)
            node_leaves = (node_npad_rg >= fnode_idx[:, :, :, :, :1]) ^ (node_npad_rg > fnode_idx[:, :, :, :, 1:])
            node_leaves = node_leaves.type_as(weights)
            node_npad = torch.tril(torch.matmul(node_leaves, node_leaves.transpose(3, 4)).clamp_(0, 1)).type_as(leave_mask)
            # node_mask:                [b, h, m, nk, nk]
            node_npad = node_npad.view(bsz, num_heads, m, tq, 1, tq).transpose(2, 3)
            node_mask = (~node_npad) | node_pad
        return leave_mask, node_mask

    @classmethod
    def build_subtree_n_leaves_mask(
            cls, num_heads, weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, m, **kwargs):
        with torch.no_grad():
            device = node_indices.device
            key_pad = key_pad if key_pad is not None else torch.zeros(bsz, m, tk, device=device).byte()
            node_pad = node_pad if node_pad is not None else torch.zeros(bsz, m, nk, device=device).byte()
            # b, h, t, m__, m_, tknk = weights.size()
            assert tq == nk, f'{tq} != {nk}.. only apply to nodes'
            key_pad = key_pad.view(bsz, 1, 1, 1, m, tk)
            node_pad = node_pad.view(bsz, 1, 1, 1, m, nk)
            indices = node_indices.contiguous().view(bsz, num_heads, m, nk, 1, 1, 2).transpose(2, 3)
            leave_range = torch.arange(0, tk, dtype=indices.dtype, device=indices.device)
            # indices:                  [b, h, t, m, 1, 1, 2]

            leave_mask = key_pad.expand(bsz, num_heads, tq, m, m, tk)
            # leave_mask:               [b, h, t, m, m, tk]

            # todo: node_mask:          [b, h, nk, m, m, nk]
            node_npad_rg = leave_range.view(1, 1, 1, 1, tk)
            fnode_idx = node_indices.view(bsz, num_heads, m, nk, 2)
            node_leaves = (node_npad_rg >= fnode_idx[:, :, :, :, :1]) ^ (node_npad_rg > fnode_idx[:, :, :, :, 1:])
            node_leaves = node_leaves.type_as(weights)
            node_npad = torch.tril(torch.matmul(node_leaves, node_leaves.transpose(3, 4)).clamp_(0, 1)).type_as(
                leave_mask)
            # node_mask:                [b, h, m, nk, nk]
            node_npad = node_npad.view(bsz, num_heads, m, tq, 1, tq).transpose(2, 3)
            node_mask = (~node_npad) | node_pad

            node_mask = node_mask.expand(bsz, num_heads, tq, m, m, nk)
        return leave_mask, node_mask

    @classmethod
    def mask_subtree_only(cls, self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        # weights:                  [b, h, t, m, m, tk + nk]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        # node_indices:             [b * h, m, nk, 2]

        # out:                      [b, h, t, m, m, tk + nk]
        # leave_mask:               [b, h, t, m, m, tk]
        # fixme: should only apply to nodes:
        b, h, t, m__, m_, tknk = weights.size()
        assert t == nk, f'{t} != {nk}.. only apply to nodes'
        leave_mask, node_mask = WeightMask.build_subtree_mask(
            self.num_heads, weights, key_pad, node_pad, node_indices, bsz, t, tk, nk, m)
        # leave_mask:               [b, h, t, m, m, tk]
        # node_mask:                [b, h, t, m, 1, nk]
        pad_mask = torch.cat([leave_mask, node_mask], -1)
        src_lens_denom = (~pad_mask).type_as(weights).sum(dim=[-1, -2], keepdim=True).clamp_(min=1.0, max=10000)
        weights = self.maybe_norm_src_len(weights, src_lens_denom)
        weights = weights.float().masked_fill(pad_mask, float('-inf')).type_as(weights)
        return weights

    @classmethod
    def mask_subtree_n_leaves(cls, self, weights, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        b, h, t, m__, m_, tknk = weights.size()
        assert t == nk, f'{t} != {nk}.. only apply to nodes'
        leave_mask, node_mask = WeightMask.build_subtree_n_leaves_mask(
            self.num_heads, weights, key_pad, node_pad, node_indices, bsz, t, tk, nk, m)
        # leave_mask:               [b, h, t, m, m, tk]
        # node_mask:                [b, h, t, m, 1, nk]
        # print(f'{leave_mask.size()}, {node_mask.size()}')
        pad_mask = torch.cat([leave_mask, node_mask], -1)
        src_lens_denom = (~pad_mask).type_as(weights).sum(dim=[-1, -2], keepdim=True).clamp_(min=1.0, max=10000)
        weights = self.maybe_norm_src_len(weights, src_lens_denom)
        weights = weights.float().masked_fill(pad_mask, float('-inf')).type_as(weights)
        return weights

    @classmethod
    def mask_mutual_ancestors(
            cls, mutual_level, self, weights, key_pad, node_pad, node_idx, bsz, tk, nk, m, **kwargs):
        # weights:                  [b, h, t, m, m, tk + nk]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        num_heads = self.num_heads
        with torch.no_grad():
            device = node_idx.device
            idx_dtype = node_idx.dtype

            b, h, t, m__, m_, tknk = weights.size()
            key_pad = key_pad if key_pad is not None else torch.zeros(bsz, m, tk, dtype=torch.uint8, device=weights.device)
            node_pad = node_pad if node_pad is not None else torch.zeros(bsz, m, nk, device=device).byte()
            assert t == tk, f'{t} != {tk}.. only apply to leaves'
            assert h == num_heads
            assert b == bsz
            n = nk
            # todo: process for leaves
            # fixme: doit again: tk => t, nk ->n
            # before_collaps:       [b, h, t, m, m, n, t]
            # after_collaps:        [b, h, t, m, m, t]
            # q_le_rg:              [1, 1, t, 1, 1, 1, 1]
            # q_mask_in:            [b, h, t, 1, m, n, 1]
            # new_q_mask_in:        [b, h, t, 1, m, n, 1]
            # k_mask_in:            [b, h, 1, 1, m, n, t]
            # leave_mask_in:        [b, h, t, 1, m, t]
            le_rg = torch.arange(0, t, dtype=idx_dtype, device=device)

            idx = node_idx.contiguous().view(b, h, 1, 1, m, n, 1, 2)
            q_le_rg = le_rg.view(1, 1, t, 1, 1, 1, 1)
            k_le_rg = le_rg.view(1, 1, 1, 1, 1, 1, t)
            mask_node_pad = (~node_pad).contiguous().view(b, 1, 1, 1, m, n, 1)
            q_mask_in = (q_le_rg >= idx[:, :, :, :, :, :, :, 0]) ^ (q_le_rg > idx[:, :, :, :, :, :, :, 1])
            q_mask_in = q_mask_in * mask_node_pad

            threshold = (torch.cumsum(q_mask_in.int(), dim=5) <= mutual_level) * q_mask_in
            new_q_mask_in = torch.cumsum(torch.flip(threshold.int(), [5]), dim=5).clamp_(0, 1)
            new_q_mask_in = torch.flip(new_q_mask_in, [5]).clamp_(0, 1).byte() * q_mask_in

            k_mask_in = (k_le_rg >= idx[:, :, :, :, :, :, :, 0]) ^ (k_le_rg > idx[:, :, :, :, :, :, :, 1])
            k_mask_in = k_mask_in * mask_node_pad

            leave_mask_in = (new_q_mask_in * k_mask_in).int().max(dim=5)[0].byte()

            out_leave_pad = (~leave_mask_in) | key_pad.transpose(1, 2).view(b, 1, t, 1, m, 1)

            # todo: process for nodes
            # after:                [b, h, t, m, m, n]
            # k_no_rg:              [1, 1, 1, 1, 1, 1, t]
            # no_idx:               [b, h, 1, 1, m, n, 1, 2]
            # no_mask_in:           [b, h, 1, 1, m, n, t]
            # no_mask_in_fl:        [b, h, 1, 1, m, n, n]
            # filter_no_mask_in:    [b, h, t, 1, m, n, n]
            # out_no_mask_in:       [b, h, t, 1, m, n]
            k_no_rg = le_rg.view(1, 1, 1, 1, 1, 1, t)
            no_idx = node_idx.contiguous().view(b, h, 1, 1, m, n, 1, 2)
            no_mask_in = (k_no_rg >= no_idx[:, :, :, :, :, :, :, 0]) ^ (k_no_rg > no_idx[:, :, :, :, :, :, :, 1])
            no_mask_node_pad = (~node_pad).contiguous().view(b, 1, 1, 1, m, n, 1)
            no_mask_in = no_mask_in * no_mask_node_pad

            no_mask_in_fl = no_mask_in.type_as(weights)
            new_no_mask_in = torch.tril(torch.matmul(
                no_mask_in_fl, no_mask_in_fl.transpose(5, 6)).clamp_(0, 1)).byte()

            filter_no_mask_in = new_no_mask_in * new_q_mask_in
            out_no_mask_in = filter_no_mask_in.int().max(dim=6)[0].byte()

            out_node_pad = (~out_no_mask_in) | node_pad.view(b, 1, 1, 1, m, n)

            # out_leave_pad:      [b, h, t, 1, m, t]
            # out_node_pad:       [b, h, t, 1, m, n]
            pad_mask = torch.cat([out_leave_pad, out_node_pad], -1)

        weights = weights.float().masked_fill(pad_mask, float('-inf')).type_as(weights)
        return weights


class HierarchicalEmbedding(nn.Embedding):
    """This module learns HierarchicalEmbedding positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, args, head_dim, num_heads, max_horiz=100, max_ver=1024):
        self.args = args
        self.take_full_dim = getattr(args, 'take_full_dim', False)
        self.num_heads = num_heads
        self.onnx_trace = False
        assert head_dim // 2 == head_dim / 2, f'require even dim{head_dim}'
        self.full_dim = head_dim ** num_heads
        self.head_dim = head_dim
        self.horizontal_dim = self.vertical_dim = (self.full_dim if self.take_full_dim else head_dim) // 2

        self.max_horiz = max_horiz
        self.max_ver = max_ver
        self.num_embeddings = 1 + self.max_horiz + 1 + self.max_ver
        self.padding_idx = 0
        super().__init__(self.num_embeddings, self.horizontal_dim, self.padding_idx)

    def proj_ver_index(self, x):
        return x + 1 + self.max_horiz

    def extra_repr(self):
        return 'h={},d_horiz={},d_ver={},m_horiz={},m_ver={},nume={},full={}'.format(
            self.num_heads, self.horizontal_dim, self.vertical_dim, self.max_horiz,
            self.max_ver, self.num_embeddings, self.take_full_dim
        )

    def hori_forward(self, x):
        return super().forward(x)

    def ver_forward(self, x):
        return super().forward(x)

    def forward(self, mask, incremental_state=None):
        assert incremental_state is None
        # mask:                 [b * h, m, t ,n, 1]
        # leaves:               [b * h, m, t, d]
        # idx:                  [b * h, m, n, 2]
        # indices:              [b * h, m, 1 ,n, 2]
        # mask:                 [b * h, m, t ,n]
        # embed:                [b * h, m, t, n, d]
        # bh, m, t, d = leaves.size()
        # bh_, m, n, di = idx.size()
        bh, m, t, n, di = mask.size()
        # assert bh == bh_, f'{bh} != {bh_}'
        assert di == 1, f'{di} != 1'
        b = bh // self.num_heads
        device = mask.device

        with torch.no_grad():
            # todo: build the indices: no need gradients
            # indices = torch.flip(idx, [2]).unsqueeze_(2)
            #
            # arange = torch.arange(0, t, device=device, dtype=idx.dtype).view(1, 1, t, 1)
            # left_mask = arange >= indices[:, :, :, :, 0]
            # right_mask = arange > indices[:, :, :, :, 1]
            # mask = left_mask ^ right_mask
            mask = torch.flip(mask, [3]).squeeze_(-1)
            if self.take_full_dim:
                mask = mask.view(b, self.num_heads, m, t, n)[:, 0]

            int_mask = mask.int()
            horiz_index = torch.cumsum(int_mask, dim=2).clamp_(0, self.max_horiz).long()
            ver_index = torch.cumsum(int_mask, dim=3).clamp_(0, self.max_ver).long()

            # todo: flip it back and do gradients embedding
            horiz_index = torch.flip(horiz_index, [3])
            ver_index = torch.flip(ver_index, [3])
            ver_index = self.proj_ver_index(ver_index)

        if self.take_full_dim:
            hori_embed = self.hori_forward(horiz_index).view(b, m * t * n, self.num_heads,
                                                             self.horizontal_dim // self.num_heads)
            ver_embed = self.ver_forward(ver_index).view(b, m * t * n, self.num_heads,
                                                         self.vertical_dim // self.num_heads)
            hier_embed = torch.cat([hori_embed, ver_embed], dim=-1).transpose(1, 2)
            hier_embed = hier_embed.contiguous().view(bh, m, t, n, self.head_dim)
        else:
            hori_embed = self.hori_forward(horiz_index)
            ver_embed = self.ver_forward(ver_index)
            hier_embed = torch.cat([hori_embed, ver_embed], dim=-1)
        return hier_embed


class HierarchicalHozSinusoidalEmbedding(HierarchicalEmbedding):

    def __init__(self, args, head_dim, num_heads, max_horiz=100, max_ver=1024):
        self.args = args
        self.take_full_dim = getattr(args, 'take_full_dim', False)
        self.num_heads = num_heads
        self.onnx_trace = False
        assert head_dim // 2 == head_dim / 2, f'require even dim{head_dim}'
        self.full_dim = head_dim ** num_heads
        self.head_dim = head_dim
        self.horizontal_dim = self.vertical_dim = (self.full_dim if self.take_full_dim else head_dim) // 2

        self.max_horiz = max_horiz
        self.max_ver = max_ver
        self.num_embeddings = 1 + self.max_horiz + 1 + self.max_ver
        self.padding_idx = 0
        nn.Embedding.__init__(self, self.num_embeddings, self.horizontal_dim, self.padding_idx)


class TopDownHierarchicalEmbedding(HierarchicalEmbedding):

    def forward(self, mask, incremental_state=None):
        assert incremental_state is None
        # mask:                 [b * h, m, t ,n, 1]

        # leaves:               [b * h, m, t, d]
        # idx:                  [b * h, m, n, 2]
        # indices:              [b * h, m, 1 ,n, 2]
        # mask:                 [b * h, m, t ,n]
        # embed:                [b * h, m, t, n, d]
        # bh, m, t, d = leaves.size()
        # bh_, m, n, di = idx.size()
        bh, m, t, n, di = mask.size()
        # assert bh == bh_, f'{bh} != {bh_}'
        assert di == 1, f'{di} != 1'
        b = bh // self.num_heads
        device = mask.device

        with torch.no_grad():
            # todo: build the indices: no need gradients
            mask = torch.flip(mask, [3]).squeeze_(-1)

            int_mask = mask.int()
            horiz_index = torch.cumsum(int_mask, dim=2).clamp_(0, self.max_horiz).long()
            ver_index = torch.cumsum(int_mask, dim=3).clamp_(0, self.max_ver).long()
            # todo: flip it back and do gradients embedding
            horiz_index = torch.flip(horiz_index, [3])

            ver_index = torch.flip(ver_index, [3])
            ver_index = self.proj_ver_index(ver_index)

        hori_embed = super().forward(horiz_index)
        ver_embed = super().forward(ver_index)

        hier_embed = torch.cat([hori_embed, ver_embed], dim=-1)
        return hier_embed


def get_hoz_bin_redo(indices):
    from collections import deque
    cache = [indices[0]]
    embed = [0]
    cur_embed = 0
    for i, x in enumerate(indices[1:]):
        first = cache[0]
        if x[0] >= first[0] and x[1] < first[1]:
            embed.append(cur_embed)
            cur_embed += 1
            cache.append(x)
        elif x[1] == first[1]:
            embed.append(cur_embed)
            cur_embed = 0
            cache.append(x)
            cache = cache[1:]
        else:
            while not (x[0] >= first[0] and x[1] <= first[1]):
                cache = cache[1:]
                first = cache[0]
            cur_embed = 0
            embed.append(cur_embed)
            cur_embed += 1
            cache.append(x)
    return embed


def get_hoz_pt(indices):
    # [b, t, 2]
    b, t, _ = indices.size()
    embed = indices.new(b, t)
    cache_idx = torch.arange(0, b).int() * t
    cur_embed = torch.zeros(b).int()
    fl_indices = indices.view(-1, 2)
    for i in range(t):
        slice = indices[:, i]
        # slice: [b, 2]
        first = fl_indices.index_select(0, cache_idx)

        first_cond = (slice[0] >= first[0]) & (slice[1] < first[1])
        second_cond = (slice[1] == first[1])

# x = [[0, 4], [0, 1], [2, 4], [3, 4]]
# x = [[0, 4], [0, 1], [2, 4], [2, 2], [3, 4]]
# x = [[0,10],[0,6],[7,10],[0,4],[5,6],[7,8],[9,10],[0,2],[3,4]]
#
# x = [[0,12],[0,12],[0,6],[7,12],[0,4],[5,6],[7,8],[9,10],[11,12],[0,2],[3,4]]
# zz
# print(get_hoz_bin(x))

class _StackNodesAttention(nn.Module):
    WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']

    def __init__(
            self, args, embed_dim, num_heads, dropout=0., bias=True,
            add_bias_kv=False, add_zero_attn=False, padding_idx=1, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.onnx_trace = False
        self.padding_idx = padding_idx

        self.src_len_norm = getattr(args, 'src_len_norm', 'none')
        self.cum_node = getattr(args, 'cum_node', 'sum')
        self.nstack_on = getattr(args, 'nstack_on', 'value')
        self.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
        self.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
        self.nstack_linear = getattr(args, 'nstack_linear', False)

        self.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)

        self.wnstack_norm = kwargs.get('wnstack_norm', getattr(args, 'wnstack_norm', 'none'))
        self.wnstack_up_norm = kwargs.get('wnstack_up_norm', getattr(args, 'wnstack_up_norm', 'none'))
        self.nstack_mask_fname = kwargs.get('nstack_mask_fn', getattr(args, 'nstack_mask_fn', 'default'))

        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)
        self.nstack_mask_func = WeightMask.acquire_wmask_function(self.nstack_mask_fname, self.mutual_level)

        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

        # max_horiz=100, max_ver=1024

        print(f'Acquire Mask function[{self.nstack_mask_fname}]: {self.nstack_mask_func}')

        assert self.wnstack_norm in self.WNSTACK_NORM
        assert self.nstack_on in ['value', 'key'], f'{self.nstack_on}'
        self._build_layer()

    def extra_repr(self):
        return 'sln={},on={},posemb={},posemb_l={},hier_emb={},cumnode={},linear={},wfname={},upnorm={},hier_share={}'.format(
            self.src_len_norm, self.nstack_on, self.nstack_pos_embed, self.nstack_pos_embed_learned,
            self.nstack_hier_embed, self.cum_node,
            self.nstack_linear, self.nstack_mask_fname, self.wnstack_up_norm, self.nstack_hier_embed_share
        )

    def get_hier_embed_positions(self):
        if not self.nstack_hier_embed:
            return None
        inside_embed = getattr(self.args, 'ShareHierarchicalEmbedding', None)
        if self.nstack_hier_embed_share and inside_embed is not None:
            # inside_embed = getattr(self.args, 'ShareHierarchicalEmbedding', None)
            print(f'Found share ShareHierarchicalEmbedding')
            return inside_embed
        else:
            print(f'Create new ShareHierarchicalEmbedding')
            new_embed = HierarchicalEmbedding(
                self.args, self.head_dim, self.num_heads,
                self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver)
            if self.nstack_hier_embed_share:
                print(f'Set embed to args')
                self.args.ShareHierarchicalEmbedding = new_embed
            return new_embed

    def _build_layer(self):
        self.src_len_norm = getattr(self.args, 'src_len_norm', 'sqrt')

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = self.add_zero_attn

        self.nstack_linear_layer = NstackLinear(self.head_dim, self.head_dim, False) if self.nstack_linear else None

        self.hier_embed_positions = self.get_hier_embed_positions()

        self.embed_positions = PositionalEmbedding(
            self.args.max_source_positions, self.head_dim, self.padding_idx,
            left_pad=False,
            learned=self.nstack_pos_embed_learned,
        ) if self.nstack_pos_embed else None

        assert not (self.hier_embed_positions is not None and self.embed_positions is not None)

        self.reset_parameters()

        self.onnx_trace = False

    def maybe_norm_src_len(self, attn_weights, src_lens_denom):
        if self.src_len_norm == 'sqrt':
            return attn_weights / src_lens_denom.sqrt()
        elif self.src_len_norm == 'none':
            return attn_weights
        elif self.src_len_norm == 'intact':
            return attn_weights / src_lens_denom
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

    @property
    def recusive_params(self):
        # FIXME: not sure about bias for now, ommit the bias
        # todo: here return param for value
        if not self.nstack_linear:
            return []

        weight = self.nstack_linear_layer.weight
        return [weight]

    def prepare_dptree_qkv(
            self, query, key, value, node_key, node_value, nsent, key_padding_mask=None, node_padding_mask=None,
            incremental_state=None, need_weights=True, static_kv=False,
            compute_query_nodes=True,
            compute_key_nodes=True,
            force_self_att=False):

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        node_kv_same = node_key.data_ptr() == node_value.data_ptr()

        tgt_len, query_bsz, embed_dim = query.size()
        leave_len, key_bsz, embed_dim_ = key.size()
        node_len, key_bsz, embed_dim_ = node_key.size()

        bsz = key_bsz // nsent
        assert nsent * bsz == key_bsz
        tk = leave_len
        nk = node_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, query_bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = node_key = node_value = None
        else:
            saved_state = None

        # query = x.transpose(1, 2).contiguous().view((tq + nq) * nsent, bsz, dim)
        if not compute_query_nodes:
            ori_query = query
            ex_query = query.view(tk + nk, nsent, query_bsz, embed_dim)
            leave_query = ex_query[:tk].contiguous().view(tk * nsent, query_bsz, embed_dim)
            node_query = ex_query[tk:]

            query = leave_query

        if qkv_same or force_self_att:
            # self-attention
            # q, k, v = self.in_proj_qkv(query)
            q = self.in_proj_q(query)
            assert node_kv_same
            k, v = self.in_proj_kv(key)
            if compute_key_nodes:
                node_k, node_v = self.in_proj_kv(node_key)
            else:
                node_k = node_v = node_key
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
                node_k = node_v = None
            else:
                k, v = self.in_proj_kv(key)
                assert node_kv_same

                if compute_key_nodes:
                    node_k, node_v = self.in_proj_kv(node_key)
                else:
                    node_k = node_v = node_key
        else:
            raise NotImplementedError(f'free-style attention not yet')

        if not compute_query_nodes:
            q = q.view(tk, nsent, query_bsz, embed_dim)
            q = torch.cat([q, node_query], 0).view((tk + nk) * nsent, query_bsz, embed_dim)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            raise NotImplementedError

        # print(f'prep: {q.size()}, k{k.size()}, nk{node_k.size()}, v{v.size()}, nv{node_v.size()}')

        q = q.contiguous().view(tgt_len, query_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if node_k is not None:
            node_k = node_k.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if node_v is not None:
            node_v = node_v.contiguous().view(-1, key_bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            assert static_kv, f'static_kv={static_kv}, only cross-attention impl here'

            def maybe_concat_state_kv(name, x):
                if name in saved_state:
                    prev = saved_state[name]
                    try:
                        prev = prev.view(bsz * nsent * self.num_heads, -1, self.head_dim)
                    except RuntimeError as er:
                        print(f'prev: {prev.size()}')
                        raise er
                    if static_kv:
                        o = prev
                    else:
                        o = torch.cat((prev, x), dim=1)
                else:
                    o = x
                saved_state[name] = o.view(bsz, nsent, self.num_heads, -1, self.head_dim)
                return o

            k = maybe_concat_state_kv('prev_key', k)
            v = maybe_concat_state_kv('prev_value', v)
            node_k = maybe_concat_state_kv('prev_node_key', node_k)
            node_v = maybe_concat_state_kv('prev_node_value', node_v)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        node_src_len = node_k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(-1) == src_len
        if node_padding_mask is not None:
            assert node_padding_mask.size(-1) == node_src_len

        if self.add_zero_attn:
            raise NotImplementedError

        return q, k, v, node_k, node_v, key_padding_mask, node_padding_mask, saved_state, src_len, node_src_len, tgt_len, query_bsz

    def get_pos_embed(self, mask):
        """

        :param mask:                [b, m, t, n, 1]
        :return: embeds:            [b, m, t, n, c]
        """
        b, m, t, n, d = mask.size()
        assert d == 1
        if self.embed_positions is not None:
            fl_mask = mask.long().squeeze(-1).transpose(2, 3).contiguous().view(b * m * n, t)
            fl_mask += self.embed_positions.padding_idx

            embeds = self.embed_positions(fl_mask)
            emb_dim = embeds.size(-1)
            embeds = embeds.view(b, m, n, t, emb_dim).transpose(2, 3)
            return embeds
        elif self.hier_embed_positions is not None:
            embeds = self.hier_embed_positions(mask)
            return embeds
        else:
            return 0.0

    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        float_mask = mask.type_as(nodes)
        rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumprod(stack, dim=3)
        assert not torch.isinf(upward_cum).any(), f'{nodes.max()}-{nodes.min()}=={leaves.max()}-{leaves.min()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states

    def accumulate_rightward(self, node_states, mask, leaves=None, **kwargs):
        """

        :param node_states:         [b, m, t, n, c]
        :param mask:                [b, m, t, n, 1]
        :param leaves:
        :param kwargs:
        :return:
        """
        if self.cum_node == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000)
            node_sum = node_states.sum(dim=2)
            rv_node_out = node_sum / mask_length
            assert not torch.isnan(
                rv_node_out).any(), f'rv_node_out: nan? {mask_length.max()} - {mask_length.min()} - {node_sum.max()} {node_sum.min()} - {node_states.max()} {node_states.min()}'
        elif self.cum_node == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000).sqrt_()
            node_sum = node_states.sum(dim=2)
            rv_node_out = node_sum / mask_length
            assert not torch.isnan(
                rv_node_out).any(), f'rv_node_out: nan? {mask_length.max()} - {mask_length.min()} - {node_sum.max()} {node_sum.min()} - {node_states.max()} {node_states.min()}'
        elif self.cum_node == 'sum':
            rv_node_out = node_states.sum(dim=2)
        else:
            raise NotImplementedError(f'Wrong cum_node={self.cum_node}')

        return rv_node_out

    def _compute_nstree_states(self, leaves, rv_nodes, indices, **kwargs):
        """
            ### nodes need to be reverse! ---> last node --> root
            ### So first is leaves, last is roots!
        :param leaves:              [b, m, t, c]
        :param rv_nodes:            [b, m, n, c]
        :param indices:             [b, m, n, 2]
        :param kwargs:
        :return:
        """
        b, m, t, c = leaves.size()
        b_, m_, n, c_ = rv_nodes.size()
        b__, m__, n_, _ = indices.size()
        assert b == b_, f'{b} != {b_}'
        assert m == m_, f'{m} != {m_}'
        assert c == c_, f'{c} != {c_}'
        assert n_ == n, f'{n} != {n_}'
        device = leaves.device
        leaves = leaves.unsqueeze(3)
        indices = indices.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(2)
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        arange = torch.arange(0, t, device=device, dtype=indices.dtype).view(1, 1, t, 1, 1)
        left_mask = arange >= indices[:, :, :, :, :1]
        right_mask = arange > indices[:, :, :, :, 1:]
        mask = left_mask ^ right_mask

        pos_embed = self.get_pos_embed(mask)
        nodes = rv_nodes + pos_embed
        assert not torch.isnan(rv_nodes).any()

        node_states = self.accumulate_upward(leaves, nodes, mask, **kwargs)

        assert not torch.isnan(node_states).any()
        assert not torch.isinf(node_states).any()
        rv_node_out = self.accumulate_rightward(node_states, mask, leaves, **kwargs)
        assert not torch.isnan(rv_node_out).any()

        return rv_node_out

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)
        attn_weights = torch.bmm(q, keys.transpose(1, 2))

        node_v = self._compute_nstree_states(v, node_v, indices, **kwargs)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values

    def mask_weights(self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        return self.nstack_mask_func(
            self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs
        )

    def compute_nstack_att(
            self, q, k, v, node_k, node_v, indices, key_pad, node_pad, src_len, tgt_len, node_src_len,
            bsz, need_weights, **kwargs
    ):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:         [b, m, tk]
        :param node_pad:        [b, m, nk]
        :param src_len:
        :param tgt_len:
        :param node_src_len:
        :param bsz:
        :param need_weights:
        :return:
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        attn_weights, values = self.compute_att_weights(
            q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs
        )

        # att_weights:          [b * h, tq, m * (tk + nk)]

        assert not torch.isnan(attn_weights).any(), f'attn_weights: before mask'
        attn_weights = self.mask_weights(attn_weights, key_pad, node_pad, indices, bsz, tq, tk, nk, nsent)
        assert not torch.isnan(attn_weights).any(), f'attn_weights: after mask'
        assert not torch.isnan(values).any() and not torch.isinf(values).any(), f'V[{values.max()}, {values.min()}]:::'

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # assert not torch.isnan(attn_weights).any(), f'attn_weights: after Softmax NaN'
        # Since some docs have empty tree in batch, softmax(-inf all) -> NaN -> replace with zeros
        # attn_weights[attn_weights != attn_weights] = 0
        # if torch.isnan(attn_weights).any():
        #     print(f'WARNING: softmax-attn_weights Nan -> zeros')
        if attn_weights.dtype == torch.float16:
            attn_weights[torch.isnan(attn_weights)] = 0.0
        else:
            attn_weights = torch.where(torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights)

        # attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        # attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_weights = self.dropout_layer(attn_weights)
        assert not torch.isnan(attn_weights).any(), f'attn_weights: after Softmax Dropoute, before bmm'

        attn = torch.bmm(attn_weights, values)

        # FIXME: damn it, I miss this line
        # out_proj

        assert not torch.isnan(attn).any(), f'attn[{attn_weights.min()},{attn_weights.max()}]: {attn_weights}'

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        assert not self.onnx_trace
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)

        attn = self.out_proj(attn)

        # print(f'att:need_weights={need_weights}')

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, att_nelems)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:

            attn_weights = None

        return attn, attn_weights

    def forward(
            self, query, key, value,
            node_key, node_value,
            indices,
            key_padding_mask=None,
            node_padding_mask=None,
            incremental_state=None,
            compute_query_nodes=True,
            compute_key_nodes=True,
            need_weights=True, static_kv=False, attn_mask=None, force_self_att=False):
        """

        :param query:               [tq, b, c]
        :param key:                 [tk, b, m, c]
        :param value:               [tk, b, m, c]
        :param node_key:            [nk, b, m, c]
        :param node_value:          [nk, b, m, c]
        :param indices:             [nk, b, m, 2]
        :param key_padding_mask:    [b, m, tk]
        :param node_padding_mask:   [b, m, nk]
        :param incremental_state:
        :param need_weights:
        :param static_kv:
        :param attn_mask:
        :param force_self_att:
        :return:
        """
        # print(f'need_weights={need_weights}')

        tq, bsz, dim = query.size()
        tk, bsz_k, nsent, dim_k = key.size()
        nk, bsz_k_, nsent_, dim_k_ = node_key.size()
        nk_, bsz_i, nsent__, idim = indices.size()

        assert bsz == bsz_k
        assert bsz_k_ == bsz_k
        assert nk == nk_, f'{node_key.size()} != {indices.size()}'
        assert idim == 2
        if key_padding_mask is None:
            assert node_padding_mask is None

        assert attn_mask is None, f'not None attn_mask (decoder self-attention) not ready'

        kwargs = {
            'ori_q': query,
            'ori_key': key,
            'ori_value': value,
            'ori_node_key': node_key,
            'ori_node_value': node_value,
            'ori_key_pad_mask': key_padding_mask,
            'ori_node_pad_mask': node_padding_mask,
        }

        f_key = key.view(tk, bsz_k * nsent, dim_k)
        f_value = value.view(tk, bsz_k * nsent, dim_k)

        f_node_key = node_key.view(nk, bsz_k * nsent, dim_k)
        f_node_value = node_value.view(nk, bsz_k * nsent, dim_k)

        # f_key_pad_mask = key_padding_mask = key_padding_mask.view(key_bsz * nsent, tk)
        assert not torch.isnan(query).any()

        (q, f_key, f_value, f_node_key, f_node_value, key_padding_mask, node_padding_mask,
         saved_state, src_len, node_src_len, tgt_len, query_bsz) = self.prepare_dptree_qkv(
            query, f_key, f_value, f_node_key, f_node_value,
            nsent,
            key_padding_mask,  node_padding_mask,
            incremental_state, need_weights, static_kv,
            compute_query_nodes=compute_query_nodes, compute_key_nodes=compute_key_nodes, force_self_att=force_self_att
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

        f_key = f_key.view(bsz_k, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_key = f_key.view(bsz_k * self.num_heads, nsent, tk, self.head_dim)

        f_value = f_value.view(bsz_k, nsent, self.num_heads, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_value = f_value.view(bsz_k * self.num_heads, nsent, tk, self.head_dim)

        f_node_key = f_node_key.view(bsz_k, nsent, self.num_heads, nk, self.head_dim).permute(0, 2, 1, 3,
                                                                                              4).contiguous()
        f_node_key = f_node_key.view(bsz_k * self.num_heads, nsent, nk, self.head_dim)

        f_node_value = f_node_value.view(bsz_k, nsent, self.num_heads, nk, self.head_dim).permute(0, 2, 1, 3,
                                                                                                  4).contiguous()
        f_node_value = f_node_value.view(bsz_k * self.num_heads, nsent, nk, self.head_dim)

        f_indices = indices.permute(1, 2, 0, 3).unsqueeze(1)
        f_indices = f_indices.expand(bsz_k, self.num_heads, nsent, nk, 2).contiguous().view(
            bsz_k * self.num_heads, nsent, nk, 2)

        # q:        [bq * h, tq, d]
        # fk:       [bk * h, m, nk, d]
        # fnk:      [bk * h, m, nk, d]
        # idx:      [bk * h, m, nk, 2]

        # fpad: [bk, m, tk]
        # fpad: [bk, m, nk]
        kwargs['after_query'] = q

        (attn, attn_weights) = self.compute_nstack_att(
            q, f_key, f_value, f_node_key, f_node_value, f_indices, key_padding_mask, node_padding_mask,
            src_len, tgt_len, node_src_len, bsz, need_weights,
            **kwargs
        )
        # tgt_len, bsz, self.embed_dim
        return attn, attn_weights


class _CSumStackNodesAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, is_affinity=False, **kwargs):
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        float_mask = mask.type_as(nodes)
        node_stack = nodes * float_mask
        if self.wnstack_include_leaves:
            stack = torch.cat([leaves, node_stack], 3)
            assert not torch.isinf(nodes).any()
            assert not torch.isinf(stack).any()
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum[:, :, :, 1:]
        else:
            stack = node_stack
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum
        if self.wnstack_up_norm == 'mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
        elif self.wnstack_up_norm == 'sqrt_mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
        node_states *= float_mask
        return node_states

    def accumulate_rightward(self, node_states, mask, leaves=None, **kwargs):
        return super().accumulate_rightward(node_states, mask, leaves, **kwargs)


class _WeightedStackNodesAttention(_StackNodesAttention):

    def accumulate_upward(self, leaves, nodes, mask, is_affinity=False, **kwargs):
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        float_mask = mask.type_as(nodes)
        node_stack = nodes * float_mask
        if self.wnstack_include_leaves:
            stack = torch.cat([leaves, node_stack], 3)
            assert not torch.isinf(nodes).any()
            assert not torch.isinf(stack).any()
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum[:, :, :, 1:]
        else:
            stack = node_stack
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum
        if self.wnstack_up_norm == 'mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
        elif self.wnstack_up_norm == 'sqrt_mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
        node_states *= float_mask
        return node_states

    def accumulate_rightward(self, node_states, mask, leaves, **kwargs):
        """

        :param node_states:         [b, m, t, n, c]
        :param mask:                [b, m, t, n, 1]
        :param leaves:              [b, m, t, 1, c]
        :param kwargs:
        :return: rv_node_out:       [b, m, n, c]
        """

        node_states = node_states.transpose(2, 4)
        leaves = leaves.permute(0, 1, 4, 2, 3)
        # nodes_states:             [b, m, c, n, t]
        # leaves:                   [b, m, c, t, 1]
        rv_node_out = torch.matmul(node_states, leaves).squeeze(-1).transpose(2, 3)
        # rv_node_out:              [b, m, n, c]

        # WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000)
            rv_node_out /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000).sqrt_()
            rv_node_out /= mask_length

        return rv_node_out


class _DistinctWeightedStackNodesAttention(_WeightedStackNodesAttention):
    """
    :param query:               [tq, b, c]
    :param key:                 [tk, b, m, c]
    :param value:               [tk, b, m, c]
    :param node_key:            [nk, b, m, c]
    :param node_value:          [nk, b, m, c]
    :param indices:             [nk, b, m, 2]
    kwargs = {
        'ori_q': query,
        'ori_key': key,
        'ori_value': value,
        'ori_node_key': node_key,
        'ori_node_value': node_value,
        'ori_key_pad_mask': key_padding_mask,
        'ori_node_pad_mask': node_padding_mask,
    }
    """

    def _build_layer(self):
        self.src_len_norm = getattr(self.args, 'src_len_norm', 'sqrt')
        self.dwstack_proj_act = getattr(self.args, 'dwstack_proj_act', 'none')

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = self.add_zero_attn

        self.nstack_linear_layer = NstackLinear(self.head_dim, self.head_dim, False) if self.nstack_linear else None

        self.dwstack_linear = transformer.Linear(self.embed_dim, self.num_heads)
        self.project_dwstack_key = lambda x: self.dwstack_linear(x)
        if self.dwstack_proj_act == 'sigmoid':
            self.project_dwstack_key = lambda x: self.dwstack_linear(x).sigmoid()
        elif self.dwstack_proj_act == 'tanh':
            self.project_dwstack_key = lambda x: self.dwstack_linear(x).tanh()

        self.hier_embed_positions = self.get_hier_embed_positions()

        self.embed_positions = PositionalEmbedding(
            self.args.max_source_positions, self.head_dim, self.padding_idx,
            left_pad=False,
            learned=self.nstack_pos_embed_learned,
        ) if self.nstack_pos_embed else None

        assert not (self.hier_embed_positions is not None and self.embed_positions is not None)

        self.reset_parameters()

        self.onnx_trace = False

    def extra_repr(self):
        prev = super().extra_repr()
        return f'{prev},dwstack_proj_act={self.dwstack_proj_act}'

    def compute_distinct_leave_states(self, leaves, **kwargs):
        """

        :param leaves:      [t, b, m, d]
        :param kwargs:
        :return:            [b * h, m, 1, t, 1]
        """
        otk, ob, om, oc = leaves.size()
        out = self.project_dwstack_key(leaves)
        # [t, b, m, h]
        out = out.permute(1, 3, 2, 0)
        out = out.contiguous().view(ob * self.num_heads, om, 1, otk, 1)
        # [b, h, m, 1, t, 1]
        return out

    def accumulate_rightward(self, node_states, mask, leaves, **kwargs):
        """

        :param node_states:         [b, m, t, n, c]
        :param mask:                [b, m, t, n, 1]
        :param leaves:              [b, m, t, 1, c]
        :param kwargs:
        :return: rv_node_out:       [b, m, n, c]
        """
        b, m, t, n, c = node_states.size()
        node_states = node_states.transpose(2, 4)
        # leaves = leaves.permute(0, 1, 4, 2, 3)

        ori_key = kwargs.get('ori_key', None)
        assert ori_key is not None
        # [tk, b, m, c]
        otk, ob, om, oc = ori_key.size()
        assert otk == t, f'{otk} != {t}'
        assert om == m, f'{om} != {m}'

        sep_leaves = self.compute_distinct_leave_states(ori_key)

        # nodes_states:             [b, m, c, n, t]
        # sep_leaves:               [b, m, 1, t, 1]
        rv_node_out = torch.matmul(node_states, sep_leaves).squeeze(-1).transpose(2, 3)
        # rv_node_out:              [b, m, n, c]

        # WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000)
            rv_node_out /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000).sqrt_()
            rv_node_out /= mask_length

        return rv_node_out


class _DistinctWeightedSplitUpDownStackNodesAttention(_DistinctWeightedStackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, is_affinity=False, **kwargs):
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]

        if is_affinity:
            float_mask = mask.type_as(nodes)
            node_stack = nodes * float_mask
            if self.wnstack_include_leaves:
                stack = torch.cat([leaves, node_stack], 3)
                assert not torch.isinf(nodes).any()
                assert not torch.isinf(stack).any()
                upward_cum = torch.cumsum(stack, dim=3)
                node_states = upward_cum[:, :, :, 1:]
            else:
                stack = node_stack
                upward_cum = torch.cumsum(stack, dim=3)
                node_states = upward_cum
            if self.wnstack_up_norm == 'mean':
                node_states /= torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
            elif self.wnstack_up_norm == 'sqrt_mean':
                node_states /= torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
            node_states *= float_mask
        else:
            float_mask = mask.type_as(nodes)
            node_stack = nodes * float_mask
            assert node_stack.size(-1) > 1, f'{node_stack.size()}, {leaves.size()}'
            assert node_stack.size(-1) == leaves.size(-1), f'{node_stack.size()}, {leaves.size()}'
            if self.wnstack_include_leaves:
                stack = torch.cat([leaves, node_stack], 3)
                first_stack, second_stack = stack.chunk(2, dim=-1)
                assert not torch.isinf(nodes).any()
                assert not torch.isinf(stack).any()
                assert not torch.isinf(first_stack).any()
                assert not torch.isinf(second_stack).any()
                upward_cum = torch.cumsum(first_stack, dim=3)

                # flipdown
                downward_cum = torch.flip(torch.cumsum(torch.flip(second_stack, [3]), dim=3), [3])

                first_node_states = upward_cum[:, :, :, 1:]
                second_node_states = downward_cum[:, :, :, 1:]
                node_states = torch.cat([first_node_states, second_node_states], -1)
            else:
                stack = node_stack
                first_stack, second_stack = stack.chunk(2, dim=-1)
                upward_cum = torch.cumsum(first_stack, dim=3)
                downward_cum = torch.flip(torch.cumsum(torch.flip(second_stack, [3]), dim=3), [3])

                # flipdown
                node_states = torch.cat([upward_cum, downward_cum], -1)

            if self.wnstack_up_norm == 'mean':
                node_states  /= torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
            elif self.wnstack_up_norm == 'sqrt_mean':
                node_states  /= torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
            node_states *= float_mask
        return node_states


class _ConvDistinctWeightedStackNodesAttention(_DistinctWeightedStackNodesAttention):
    @staticmethod
    def get_act(name):
        if name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'none':
            return None
        else:
            raise ValueError(f'{name} not valid')

    def _build_layer(self):
        self.src_len_norm = getattr(self.args, 'src_len_norm', 'sqrt')
        self.dwstack_proj_act = getattr(self.args, 'dwstack_proj_act', 'none')

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = self.add_zero_attn

        self.nstack_linear_layer = NstackLinear(self.head_dim, self.head_dim, False) if self.nstack_linear else None


        self.dwstack_conv = nn.Conv1d(self.embed_dim, self.num_heads, kernel_size=2)
        self.init_conv_layer(self.dwstack_conv)
        self.project_dwstack_key = lambda x: self.dwstack_conv(x)
        if self.dwstack_proj_act == 'sigmoid':
            self.project_dwstack_key = lambda x: self.dwstack_conv(x).sigmoid()
        elif self.dwstack_proj_act == 'tanh':
            self.project_dwstack_key = lambda x: self.dwstack_conv(x).tanh()

        self.hier_embed_positions = self.get_hier_embed_positions()

        self.embed_positions = PositionalEmbedding(
            self.args.max_source_positions, self.head_dim, self.padding_idx,
            left_pad=False,
            learned=self.nstack_pos_embed_learned,
        ) if self.nstack_pos_embed else None

        self.transition_conv = self.build_transition_fn()

        assert not (self.hier_embed_positions is not None and self.embed_positions is not None)

        self.reset_parameters()
        self.onnx_trace = False

    def build_transition_fn(self):
        self.transition_act = getattr(self.args, 'transition_act', 'none')
        self.transition_dropout = getattr(self.args, 'transition_dropout', 0.0)

        transition_conv = nn.Conv2d(
            self.embed_dim, self.embed_dim, kernel_size=(2, 1),
        )
        self.init_conv_layer(transition_conv)

        stacks = []
        if self.transition_dropout > 0:
            stacks.append(nn.Dropout(self.transition_dropout))
        stacks.append(transition_conv)
        act = self.get_act(self.transition_act)
        if act is not None:
            stacks.append(act)

        transition_fn = nn.Sequential(*stacks)

        return transition_fn

    def init_conv_layer(self, conv):
        nn.init.xavier_uniform_(conv.weight)
        try:
            nn.init.constant_(conv.bias, 0.)
        except Exception as e:
            print(f'Cannot init conv bias')

    def accumulate_upward(self, leaves, nodes, mask, is_affinity=False, **kwargs):
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        # node_states:              [b, m, t, n, c]
        float_mask = mask.type_as(nodes)
        node_stack = nodes * float_mask
        if self.wnstack_include_leaves:
            stack = torch.cat([leaves, node_stack], 3)
            assert not torch.isinf(nodes).any()
            assert not torch.isinf(stack).any()
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum[:, :, :, 1:]
        else:
            stack = node_stack
            upward_cum = torch.cumsum(stack, dim=3)
            node_states = upward_cum
        if self.wnstack_up_norm == 'mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
        elif self.wnstack_up_norm == 'sqrt_mean':
            node_states = node_states / torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
        node_states *= float_mask

        # todo: do transition convolution
        node_size = node_states.size()
        bh, m, t, n, c = node_states.size()
        b = bh // self.num_heads
        h = self.num_heads
        # print(f'node_size={node_size}')
        node_states_t = node_states.contiguous().view(b, h, m, t, n, c).permute(0, 2, 1, 5, 3, 4)
        node_states_t = node_states_t.contiguous().view(b * m, h * c, t, n)
        transit_states = self.transition_conv(node_states_t)
        # node_size = torch.Size([320, 1, 66, 65, 30]) -> torch.Size([32, 300, 65, 65])
        # print(f'node_size={node_size} -> {transit_states.size()}')
        transit_states = transit_states.view(b, m, h, c, t - 1, n).permute(0, 2, 1, 4, 5, 3)
        transit_states = transit_states.contiguous().view(b * h, m, t - 1, n, c)
        # '[320, 1, 66, 65, 30]'
        transit_states *= float_mask[:, :, 1:, :, :]
        return transit_states

    def accumulate_rightward(self, node_states, mask, leaves, **kwargs):
        """

        :param node_states:         [b, m, t, n, c]
        :param mask:                [b, m, t, n, 1]
        :param leaves:              [b, m, t, 1, c]
        :param kwargs:
        :return: rv_node_out:       [b, m, n, c]
        """
        b, m, t, n, c = node_states.size()
        node_states = node_states.transpose(2, 4)
        # leaves = leaves.permute(0, 1, 4, 2, 3)

        ori_key = kwargs.get('ori_key', None)
        assert ori_key is not None
        # [tk, b, m, c]
        otk, ob, om, oc = ori_key.size()
        assert otk - 1 == t, f'{otk} != {t}'
        assert om == m, f'{om} != {m}'

        sep_leaves = self.compute_distinct_leave_states(ori_key)

        # nodes_states:             [b, m, c, n, t]
        # sep_leaves:               [b, m, 1, t, 1]
        rv_node_out = torch.matmul(node_states, sep_leaves).squeeze(-1).transpose(2, 3)
        # rv_node_out:              [b, m, n, c]

        # WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']
        mask = mask[:, :, 1:]
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000)
            rv_node_out /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000).sqrt_()
            rv_node_out /= mask_length

        return rv_node_out

    def compute_distinct_leave_states(self, leaves, **kwargs):
        """

        :param leaves:      [t, b, m, d]
        :param kwargs:
        :return:            [b * h, m, 1, t, 1]
        """
        otk, ob, om, oc = leaves.size()
        # leaves = leaves.permute(1, 2, 3, 0).contiguous().view(ob * om, oc, otk)
        leaves = leaves.view(otk, ob * om, oc).permute(1, 2, 0)
        out = self.project_dwstack_key(leaves)
        # this_is_conv
        # [t, b, m, h]
        # [bm, h, t]
        h = out.size(1)
        out = out.contiguous().view(ob, om, h, 1, otk - 1, 1).transpose(1, 2)
        out = out.view(ob * h, om, 1, otk - 1, 1)
        # out = out.contiguous().view(ob * self.num_heads, om, 1, otk, 1)
        # [b, h, m, 1, t, 1]
        return out


class _SeparateWDistinctWeightedStackNodesAttention(_DistinctWeightedStackNodesAttention):
    def _build_layer(self):
        self.src_len_norm = getattr(self.args, 'src_len_norm', 'sqrt')
        self.dwstack_proj_act = getattr(self.args, 'dwstack_proj_act', 'none')
        self.sep_dwstack_proj_act = getattr(self.args, 'sep_dwstack_proj_act', 'tanh')

        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * self.embed_dim, self.embed_dim))
        if self.bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * self.embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.bias)

        if self.add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, self.embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, self.embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = self.add_zero_attn

        self.nstack_linear_layer = NstackLinear(self.head_dim, self.head_dim, False) if self.nstack_linear else None

        self.dwstack_linear = transformer.Linear(self.embed_dim, self.embed_dim)

        def dwstack_transform(q, k_ori, key_pad, node_pad):
            # k:        [t, b, m, d]
            # q:        [b * h, (t + n) * m, c]
            # n_q:      [b * h, m, n, c]
            # kk:       [b * h, m, c, t]
            # w:        [b * h, m, n, t]
            # key_pad   [b, m, t]
            # node_pad   [b, m, n]
            t, b, m, d = k_ori.size()
            bh, tnm, c = q.size()
            tn = tnm // m
            n = tn - t
            n_q = q.view(bh, tn, m, c)[:, t:].transpose(1, 2)

            kk = self.dwstack_linear(k_ori)
            kk = kk.contiguous().view(t, b, m, self.num_heads, self.head_dim).permute(1, 3, 2, 4, 0)
            kk = kk.view(b * self.num_heads, m, self.head_dim, t)
            w = torch.matmul(n_q, kk)
            w *= self.scaling
            if key_pad is not None:
                assert node_pad is not None
                pad_mask = key_pad.view(b, 1, m, 1, t) | node_pad.view(b, 1, m, n, 1)
                pad_mask = pad_mask.expand(b, self.num_heads, m, n, t).contiguous().view(bh, m, n, t)
                # w = w.float().masked_fill(pad_mask, float('-inf'))
                w = w * (~pad_mask).type_as(w)
            return w

        self.project_dwstack_key = dwstack_transform
        if self.sep_dwstack_proj_act == 'sigmoid':
            self.project_dwstack_key = lambda *inputs: dwstack_transform(*inputs).sigmoid()
        elif self.sep_dwstack_proj_act == 'tanh':
            self.project_dwstack_key = lambda *inputs: dwstack_transform(*inputs).tanh()

        self.hier_embed_positions = self.get_hier_embed_positions()

        self.embed_positions = PositionalEmbedding(
            self.args.max_source_positions, self.head_dim, self.padding_idx,
            left_pad=False,
            learned=self.nstack_pos_embed_learned,
        ) if self.nstack_pos_embed else None

        assert not (self.hier_embed_positions is not None and self.embed_positions is not None)

        self.reset_parameters()

        self.onnx_trace = False

    def accumulate_rightward(self, node_states, mask, leaves, **kwargs):
        """

        :param node_states:         [b, m, t, n, c]
        :param mask:                [b, m, t, n, 1]
        :param leaves:              [b, m, t, 1, c]
        :param kwargs:
        :return: rv_node_out:       [b, m, n, c]
        """
        b, m, t, n, c = node_states.size()
        node_states = node_states.transpose(2, 4)
        # leaves = leaves.permute(0, 1, 4, 2, 3)

        ori_key = kwargs.get('ori_key', None)
        after_query = kwargs.get('after_query', None)
        key_pad = kwargs.get('ori_key_pad_mask', None)
        node_pad = kwargs.get('ori_node_pad_mask', None)
        assert ori_key is not None
        assert after_query is not None
        # assert key_pad is not None
        # assert node_pad is not None
        # [tk, b, m, c]
        otk, ob, om, oc = ori_key.size()
        bh, tnm, qc = after_query.size()
        assert otk == t, f'{otk} != {t}'
        assert om == m, f'{om} != {m}'

        sep_w = self.project_dwstack_key(after_query, ori_key, key_pad, node_pad).view(bh, m, 1, n, t)
        # sep_w:                    [b, m, n, t]
        # sep_leaves = self.compute_distinct_leave_states(ori_key)

        # nodes_states:             [b, m, c, n, t]
        # sep_leaves:               [b, m, 1, t, 1]
        rv_node_out = (node_states * sep_w).sum(-1).transpose(2, 3)
        # rv_node_out = torch.matmul(node_states, sep_leaves).squeeze(-1).transpose(2, 3)
        # rv_node_out:              [b, m, n, c]

        # WNSTACK_NORM = ['none', 'mean', 'sqrt_mean']
        if self.wnstack_norm == 'mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000)
            rv_node_out /= mask_length
        elif self.wnstack_norm == 'sqrt_mean':
            mask_length = mask.type_as(node_states).sum(dim=2).clamp_(1.0, 10000).sqrt_()
            rv_node_out /= mask_length

        return rv_node_out


class NodeStackOnValueAttention(_StackNodesAttention):
    pass


class NodeStackSumUpOnValueAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        float_mask = mask.type_as(nodes)
        # rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        # node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumsum(stack, dim=3)

        # assert not torch.isinf(
        #     upward_cum).any(), f'{nodes.max()}-{nodes.min()}-{nodes.mean()}=={leaves.max()}-{leaves.min()}-{leaves.mean()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states


class NodeStackOnValueTanhUpAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        nodes = nodes.tanh()
        float_mask = mask.type_as(nodes)
        rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumprod(stack, dim=3)
        assert not torch.isinf(
            upward_cum).any(), f'{nodes.max()}-{nodes.min()}-{nodes.mean()}=={leaves.max()}-{leaves.min()}-{leaves.mean()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states


class NodeStackSumUpOnValueTanhUpAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        nodes = nodes.tanh()
        float_mask = mask.type_as(nodes)
        # rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        # node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumsum(stack, dim=3)

        # assert not torch.isinf(
        #     upward_cum).any(), f'{nodes.max()}-{nodes.min()}-{nodes.mean()}=={leaves.max()}-{leaves.min()}-{leaves.mean()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states


class NodeStackOnKeyAttention(_StackNodesAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        # if self.nstack_on == 'value':
        # nstack, stack_mask = self._extract_sntree_stack(k, node_k, indices)
        # node_k = self._compute_nstree_node_states(nstack, stack_mask)
        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


class NodeStackOnKeyTanhUpAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        nodes = nodes.tanh()
        float_mask = mask.type_as(nodes)
        rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumprod(stack, dim=3)
        assert not torch.isinf(
            upward_cum).any(), f'{nodes.max()}-{nodes.min()}-{nodes.mean()}=={leaves.max()}-{leaves.min()}-{leaves.mean()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states


class NodeStackSumUpOnKeyAttention(_StackNodesAttention):
    def accumulate_upward(self, leaves, nodes, mask, **kwargs):
        float_mask = mask.type_as(nodes)
        # rev_fl_mask = (~mask).type_as(nodes)
        node_stack = nodes * float_mask
        # node_stack += rev_fl_mask
        stack = torch.cat([leaves, node_stack], 3)
        assert not torch.isinf(nodes).any()
        assert not torch.isinf(stack).any()
        upward_cum = torch.cumsum(stack, dim=3)

        # assert not torch.isinf(
        #     upward_cum).any(), f'{nodes.max()}-{nodes.min()}-{nodes.mean()}=={leaves.max()}-{leaves.min()}-{leaves.mean()}'
        node_states = upward_cum[:, :, :, 1:]
        node_states *= float_mask
        return node_states

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        # if self.nstack_on == 'value':
        # nstack, stack_mask = self._extract_sntree_stack(k, node_k, indices)
        # node_k = self._compute_nstree_node_states(nstack, stack_mask)
        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


# FIXME: CSUm

class NodeStackCSumOnValueAttention(_CSumStackNodesAttention):
    pass


class NodeStackCSumOnKeyAttention(_CSumStackNodesAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        # if self.nstack_on == 'value':
        # nstack, stack_mask = self._extract_sntree_stack(k, node_k, indices)
        # node_k = self._compute_nstree_node_states(nstack, stack_mask)
        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


class NodeStackCSumOnKeyValueAttention(_CSumStackNodesAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        # if self.nstack_on == 'value':
        # nstack, stack_mask = self._extract_sntree_stack(k, node_k, indices)
        # node_k = self._compute_nstree_node_states(nstack, stack_mask)
        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        node_v = self._compute_nstree_states(v, node_v, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


# FIXME: weight---
class NodeStackWeightedOnValueAttention(_WeightedStackNodesAttention):
    pass


class NodeStackWeightedOnKeyAttention(_WeightedStackNodesAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


class NodeStackWeightedOnAffinityAttention(_WeightedStackNodesAttention):
    def get_pos_embed(self, mask):
        """

        :param mask:                [b, m, t, n, 1]
        :return: embeds:            [b, m, t, n, c]
        """
        b, m, t, n, d = mask.size()
        assert d == 1
        if self.embed_positions is not None:
            fl_mask = mask.long().squeeze(-1).transpose(2, 3).contiguous().view(b * m * n, t)
            fl_mask += self.embed_positions.padding_idx

            embeds = self.embed_positions(fl_mask)
            emb_dim = embeds.size(-1)
            embeds = embeds.view(b, m, n, t, emb_dim).transpose(2, 3)
            return embeds
        elif self.hier_embed_positions is not None:
            raise NotImplementedError(f'{self.__class__.__name__}: cannot used hier_embed on affinity')
            # embeds = self.hier_embed_positions(mask)
            # return embeds
        else:
            return 0.0

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2)).view(bh, tq, nsent, tk).permute(0, 2, 3,
                                                                                                              1)
        attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2)).view(bh, tq, nsent, nk).permute(0, 2,
                                                                                                                  3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        # concat_k = torch.cat([k, node_k], 2).view(bh, nsent * att_seq_len, d).transpose(1, 2)
        #
        # attn_concat_weights = torch.bmm(q, concat_k).view(bh, tq, nsent, att_seq_len).permute(0, 2, 3, 1)
        #
        # attn_leaves = attn_concat_weights[:, :, :tk]
        # attn_nodes = attn_concat_weights[:, :, tk:]
        #
        # # attn_leaves = attn_concat_weights[:, :, :, :tk].permute(0, 2, 3, 1)
        # # attn_nodes = attn_concat_weights[:, :, :, tk:].permute(0, 2, 3, 1)
        #
        # # attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2))
        # # attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2))
        # # att_leaves:           [b * h, q, m*tk]
        # # att_nodes:            [b * h, q, m*nk]
        #
        # # nstree_attn_leaves = attn_leaves.contiguous().view(bh, tq, nsent, tk).permute(0, 2, 3, 1)
        # # nstree_attn_nodes = attn_nodes.contiguous().view(bh, tq, nsent, nk).permute(0, 2, 3, 1)
        #
        # # print(f'nstree_attn_leaves: {nstree_attn_leaves.size()} - {nstree_attn_nodes.size()}')
        # # nstree_attn_nodes = self._compute_nstree_states(nstree_attn_leaves, nstree_attn_nodes, indices)
        # nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices)
        #
        # # nstree_attn_nodes:    [b * h, m, n, tq]
        # nstree_attn_nodes = nstree_attn_nodes.contiguous().view(bh, nsent * nk, tq).transpose(1, 2)
        # nstree_attn_leaves = attn_leaves.contiguous().view(bh, nsent * tk, tq).transpose(1, 2)
        #
        # # attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2)
        # attn_weights = torch.cat([nstree_attn_leaves, nstree_attn_nodes], 2)
        """
        :param leaves:              [b, m, t, c]
        :param rv_nodes:            [b, m, n, c]
        :param indices:             [b, m, n, 2]
        :param kwargs:
        """

        # node_k = self._compute_nstree_states(k, node_k, indices)
        # keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)
        # attn_weights = torch.bmm(q, keys.transpose(1, 2))

        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values


class NodeStackConvDistinctWeightedOnValueAttention(_ConvDistinctWeightedStackNodesAttention):
    pass


class NodeStackDistinctWeightedOnValueAttention(_DistinctWeightedStackNodesAttention):
    pass


class NodeStackDistinctWeightedOnKeyAttention(_DistinctWeightedStackNodesAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        node_k = self._compute_nstree_states(k, node_k, indices, **kwargs)
        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q, keys.transpose(1, 2))
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return attn_weights, values


class NodeStackDistinctWeightedOnQueryAttention(_DistinctWeightedStackNodesAttention):

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]              -> [b * h, tq, m, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tqn, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        assert tqn % nsent == 0
        tq = tqn // nsent
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        q = q.contiguous().view(bh, tq, nsent, d).transpose(1, 2)
        # q:                    [b * h, m, tq, d]
        q_leave, q_node = q.split([tk, nk], 2)

        q_node = self._compute_nstree_states(q_leave, q_node, indices, **kwargs)

        q_o = torch.cat([q_leave, q_node], 2).transpose(1, 2).contiguous().view(bh, tqn, d)

        keys = torch.cat([k, node_k], 2).view(bh, att_nelems, d)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)

        attn_weights = torch.bmm(q_o, keys.transpose(1, 2))

        # attn_weights = torch.bmm(q, keys.transpose(1, 2))
        #
        # node_v = self._compute_nstree_states(v, node_v, indices, **kwargs)
        # else:

        return attn_weights, values


class NodeStackDistinctWeightedOnAffinityAttention(_DistinctWeightedStackNodesAttention):
    def get_pos_embed(self, mask):
        """

        :param mask:                [b, m, t, n, 1]
        :return: embeds:            [b, m, t, n, c]
        """
        b, m, t, n, d = mask.size()
        assert d == 1
        if self.embed_positions is not None:
            fl_mask = mask.long().squeeze(-1).transpose(2, 3).contiguous().view(b * m * n, t)
            fl_mask += self.embed_positions.padding_idx

            embeds = self.embed_positions(fl_mask)
            emb_dim = embeds.size(-1)
            embeds = embeds.view(b, m, n, t, emb_dim).transpose(2, 3)
            return embeds
        elif self.hier_embed_positions is not None:
            raise NotImplementedError(f'{self.__class__.__name__}: cannot used hier_embed on affinity')
            # embeds = self.hier_embed_positions(mask)
            # return embeds
        else:
            return 0.0

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2)).view(
            bh, tq, nsent, tk).permute(0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2)).view(
            bh, tq, nsent, nk).permute(0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values


class NodeStackDistinctWeightedOnAffinityValueAttention(_DistinctWeightedStackNodesAttention):
    def _compute_nstree_states(self, leaves, rv_nodes, indices, is_affinity=True, **kwargs):
        """
            ### nodes need to be reverse! ---> last node --> root
            ### So first is leaves, last is roots!
        :param leaves:              [b, m, t, c]
        :param rv_nodes:            [b, m, n, c]
        :param indices:             [b, m, n, 2]
        :param kwargs:
        :return:
        """
        b, m, t, c = leaves.size()
        b_, m_, n, c_ = rv_nodes.size()
        b__, m__, n_, _ = indices.size()
        assert b == b_, f'{b} != {b_}'
        assert m == m_, f'{m} != {m_}'
        assert c == c_, f'{c} != {c_}'
        assert n_ == n, f'{n} != {n_}'
        device = leaves.device
        leaves = leaves.unsqueeze(3)
        indices = indices.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(2)
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        arange = torch.arange(0, t, device=device, dtype=indices.dtype).view(1, 1, t, 1, 1)
        left_mask = arange >= indices[:, :, :, :, :1]
        right_mask = arange > indices[:, :, :, :, 1:]
        mask = left_mask ^ right_mask

        # get_pos_embed = kwargs.get('get_pos_embed', True)
        if is_affinity:
            nodes = rv_nodes
        else:
            pos_embed = self.get_pos_embed(mask)
            nodes = rv_nodes + pos_embed

        assert not torch.isnan(rv_nodes).any()

        node_states = self.accumulate_upward(leaves, nodes, mask, is_affinity=is_affinity, **kwargs)

        assert not torch.isnan(node_states).any()
        assert not torch.isinf(node_states).any()
        rv_node_out = self.accumulate_rightward(node_states, mask, leaves, **kwargs)
        assert not torch.isnan(rv_node_out).any()

        return rv_node_out

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2)).view(bh, tq, nsent, tk).permute(0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2)).view(bh, tq, nsent, nk).permute(0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, is_affinity=True, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        node_v = self._compute_nstree_states(v, node_v, indices, is_affinity=False, **kwargs)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values


class NodeStackSepWDistinctWeightedOnAffinityAttention(_SeparateWDistinctWeightedStackNodesAttention):
    def _compute_nstree_states(self, leaves, rv_nodes, indices, is_affinity=True, **kwargs):
        """
            ### nodes need to be reverse! ---> last node --> root
            ### So first is leaves, last is roots!
        :param leaves:              [b, m, t, c]
        :param rv_nodes:            [b, m, n, c]
        :param indices:             [b, m, n, 2]
        :param kwargs:
        :return:
        """
        b, m, t, c = leaves.size()
        b_, m_, n, c_ = rv_nodes.size()
        b__, m__, n_, _ = indices.size()
        assert b == b_, f'{b} != {b_}'
        assert m == m_, f'{m} != {m_}'
        assert c == c_, f'{c} != {c_}'
        assert n_ == n, f'{n} != {n_}'
        device = leaves.device
        leaves = leaves.unsqueeze(3)
        indices = indices.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(2)
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        arange = torch.arange(0, t, device=device, dtype=indices.dtype).view(1, 1, t, 1, 1)
        left_mask = arange >= indices[:, :, :, :, :1]
        right_mask = arange > indices[:, :, :, :, 1:]
        mask = left_mask ^ right_mask

        # get_pos_embed = kwargs.get('get_pos_embed', True)
        if is_affinity:
            nodes = rv_nodes
        else:
            pos_embed = self.get_pos_embed(mask)
            nodes = rv_nodes + pos_embed

        assert not torch.isnan(rv_nodes).any()

        node_states = self.accumulate_upward(leaves, nodes, mask, is_affinity=is_affinity, **kwargs)

        assert not torch.isnan(node_states).any()
        assert not torch.isinf(node_states).any()
        rv_node_out = self.accumulate_rightward(node_states, mask, leaves, **kwargs)
        assert not torch.isnan(rv_node_out).any()

        return rv_node_out

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, m, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = m * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, m * tk, d).transpose(1, 2)).view(bh, tq, m, tk).permute(0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, m * nk, d).transpose(1, 2)).view(bh, tq, m, nk).permute(0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values


class NodeStackSepWDistinctWeightedOnAffinityValueAttention(NodeStackSepWDistinctWeightedOnAffinityAttention):
    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, m, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = m * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, m * tk, d).transpose(1, 2)).view(bh, tq, m, tk).permute(0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, m * nk, d).transpose(1, 2)).view(bh, tq, m, nk).permute(0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, is_affinity=True, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        node_v = self._compute_nstree_states(v, node_v, indices, is_affinity=False, **kwargs)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values



class NodeStackDistinctWeightedSplitUpDownOnAffinityValueAttention(_DistinctWeightedSplitUpDownStackNodesAttention):
    def _compute_nstree_states(self, leaves, rv_nodes, indices, is_affinity=True, **kwargs):
        """
            ### nodes need to be reverse! ---> last node --> root
            ### So first is leaves, last is roots!
        :param leaves:              [b, m, t, c]
        :param rv_nodes:            [b, m, n, c]
        :param indices:             [b, m, n, 2]
        :param kwargs:
        :return:
        """
        b, m, t, c = leaves.size()
        b_, m_, n, c_ = rv_nodes.size()
        b__, m__, n_, _ = indices.size()
        assert b == b_, f'{b} != {b_}'
        assert m == m_, f'{m} != {m_}'
        assert c == c_, f'{c} != {c_}'
        assert n_ == n, f'{n} != {n_}'
        device = leaves.device
        leaves = leaves.unsqueeze(3)
        indices = indices.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(2)
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        arange = torch.arange(0, t, device=device, dtype=indices.dtype).view(1, 1, t, 1, 1)
        left_mask = arange >= indices[:, :, :, :, :1]
        right_mask = arange > indices[:, :, :, :, 1:]
        mask = left_mask ^ right_mask

        # get_pos_embed = kwargs.get('get_pos_embed', True)
        if is_affinity:
            nodes = rv_nodes
        else:
            pos_embed = self.get_pos_embed(mask)
            nodes = rv_nodes + pos_embed

        assert not torch.isnan(rv_nodes).any()

        node_states = self.accumulate_upward(leaves, nodes, mask, is_affinity=is_affinity, **kwargs)

        assert not torch.isnan(node_states).any()
        assert not torch.isinf(node_states).any()
        rv_node_out = self.accumulate_rightward(node_states, mask, leaves, **kwargs)
        assert not torch.isnan(rv_node_out).any()

        return rv_node_out

    def compute_att_weights(self, q, k, v, node_k, node_v, indices, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param indices:         [b * h, m, nk, 2]
        :param key_pad:
        :param node_pad:
        :return:
            att_weights:        [b * h, tq, m * (tk + nk)]
            values:             [b * h, m * (tk + nk), d]
        """
        bh, tq, d = q.size()
        _, nsent, tk, _ = k.size()
        _, nsent_, nk, _ = node_k.size()
        att_seq_len = tk + nk
        att_nelems = nsent * att_seq_len

        attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2)).view(bh, tq, nsent, tk).permute(0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2)).view(bh, tq, nsent, nk).permute(0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(attn_leaves, attn_nodes, indices, is_affinity=True, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        node_v = self._compute_nstree_states(v, node_v, indices, is_affinity=False, **kwargs)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values


"""
Ways to improve

: cum_up:       cumprod (cumsum?)
: cum_right:    cumsum

comprod vs cumsum

@ add pos_embedding for node_tokens....
@ cumsum -> mean

@ add square weight before comprod
add square weight after comprod

@ --add_rec_param_loss >>>> add l1 loss -- 

norm how to do it correctly

upward cum: cumprod like dot product???? not element-wise product?


"""


def com(leaves, nodes, indices):
    """
    :param leaves:  [b, t, c]
    :param nodes:   [b, n, c]
    :param indices: [b, n, 2]
    :return:
    """
    b, t, c = leaves.size()
    leaves = leaves.unsqueeze(2)
    indices = indices.unsqueeze(1)
    nodes = nodes.unsqueeze(1)
    arange = torch.arange(0, t, dtype=indices.dtype).view(1, t, 1, 1)
    left_mask = arange >= indices[:, :, :, :1]
    right_mask = arange > indices[:, :, :, 1:]
    mask = left_mask ^ right_mask
    float_mask = mask.type_as(nodes)
    rev_fl_mask = (~mask).type_as(nodes)
    rv_node_stack = nodes * float_mask
    rv_node_stack = rv_node_stack + rev_fl_mask
    # leaves:                   [b, t, 1, c]
    # nodes:                    [b, 1, n, c]
    # indices:                  [b, 1, n, 2]
    # mask:                     [1, t, 1, 1]
    stack = torch.cat([leaves, rv_node_stack], 2)
    comprod = torch.cumprod(stack, dim=2)
    node_states = comprod[:, :, 1:]
    node_states *= float_mask
    rv_node_out = node_states.sum(dim=1)
    return rv_node_out, comprod, stack, rv_node_stack, mask, left_mask, right_mask


# ll = torch.tensor([[10, 9,8,7,5]]).unsqueeze_(-1)
# nn = torch.tensor([[20, 11, 23, 98]]).unsqueeze_(-1)
# indices = torch.tensor([[  [3, 4], [2, 4], [0, 1], [0, 4]     ]])
# rv_node_out, comprod, stack, rv_node_stack, mask, left_mask, right_mask = com(ll, nn, indices)
#
# stack = stack.squeeze_(-1)
# comprod = comprod.squeeze_(-1)
# mask = mask.squeeze_(-1)
# left_mask = left_mask.squeeze_(-1)
# right_mask = right_mask.squeeze_(-1)
# rv_node_stack = rv_node_stack.squeeze_(-1)


# TEsting
# import torch
#
# x = torch.tensor([
#     [1, 1, 0, 0, 0],
#     [0, 0, 0, 1, 1],
#     [0, 0, 1, 1, 1],
#     [1, 1, 1, 1, 1],
# ])
#
# print(torch.tril(torch.matmul(x, x.transpose(0, 1)).clamp_(0, 1)))
#
# x = torch.tensor([
#     []
# ])
#

# # fixme: should only apply to nodes:
# b, h, t, m__, m_, tknk = weights.size()
# assert t == nk, f'{t} != {nk}.. only apply to nodes'
# key_pad = key_pad.view(bsz, 1, 1, 1, m, tk)
# node_pad = node_pad.view(bsz, 1, 1, 1, m, nk)
#
# indices = node_indices.contiguous().view(bsz, self.num_heads, m, nk, 1, 1, 2).transpose(2, 3)
# # indices:                  [b, h, t, m, 1, 1, 2]
#
# # todo: do leave_mask
# leave_range = torch.arange(0, tk, dtype=indices.dtype, device=indices.device)
# leave_npad_range = leave_range.view(1, 1, 1, 1, 1, tk)
# leave_npad = (leave_npad_range < indices[:, :, :, :, :, :, 0]) | (leave_npad_range > indices[:, :, :, :, :, :, 1])
# # leave_pad:                [b, h, t, m, 1, tk]
# leave_mask = (~leave_npad) | key_pad
# # leave_mask:               [b, h, t, m, m, tk]
#
# # todo: node_mask:          [b, h, nk, m, m, nk]
# node_range = leave_range.view(1, 1, 1, 1, tk)
# fnode_idx = node_indices.view(bsz, self.num_heads, m, nk, 2)
# node_leaves = (node_range >= fnode_idx[:, :, :, :, :1]) ^ (node_range > fnode_idx[:, :, :, :, 1:])
# node_leaves = node_leaves.int()
#
# node_npad_mask = torch.tril(torch.matmul(node_leaves, node_leaves.transpose(3, 4)).clamp_(0, 1)).type_as(leave_mask)
# # node_mask:                [b, h, m, nk, nk]
# node_npad_mask = node_npad_mask.view(b, h, m, t, 1, t).transpose(2, 3)
#
# node_mask = node_npad_mask | node_pad

#
# x = torch.tensor([
#     [0, 10], [0, 6], [7, 10], [0, 4], [5, 6], [7, 8], [9, 10], [0, 2], [3, 4]
# ]).unsqueeze_(1)
# x = torch.flip(x, [0])
# r = torch.arange(0, 11).view(1, 11).expand(x.size(0), 11)
#
# o = (r >= x[:, :, 0]) ^ (r > x[:, :, 1])
# print(o)
#
# o = o.int()
# np = torch.tril(torch.matmul(o, o.transpose(0, 1)).clamp_(0, 1)).byte()
# print(torch.tril(torch.matmul(o, o.transpose(0, 1)).clamp_(0, 1)))

# left_mask = arange >= indices[:, :, :, :, :1]
# right_mask = arange > indices[:, :, :, :, 1:]
# mask = left_mask ^ right_mask


if __name__ == '__main__':
    print(f'Nothing here')
