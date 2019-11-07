"""
This is to build more efficient nstack_tree_attention.py
Should be optimized as much as possible,
share masking and soon
"""
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils
import os
import functools
from fairseq.models import transformer
from fairseq.modules.multihead_attention import *
from . import nstack_tree_attention as nstack_att

DEBUG = bool(int(os.environ.get('TREEDEBUG', 0)))
print(f'nstack_tree_attention_eff:debug={DEBUG}')

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


class WeightMaskEff(nstack_att.WeightMask):

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
    def acquire_wmask_with_mask_function(cls, fname, mutual_level=5):
        if fname == cls.DEFAULT:
            return cls.mask_weights_wmask
        elif fname == cls.ALL_ALL:
            return cls.mask_weights_wmask
        elif fname == cls.LEAVES_SUBTREE:
            return cls.mask_weights_leave_subtree_wmask
        elif fname == cls.LEAVES_SUBTREELEAVES:
            raise NotImplementedError(f'fname={fname}')
        elif fname == cls.LEAVESANCESTORS_SUBTREELEAVES:
            raise NotImplementedError(f'fname={fname}')
        elif fname == cls.MUTUALANCESTORS_SUBTREE:
            # return functools.partial(cls.wmask_mutualancestors_subtree, mutual_level=mutual_level)
            raise NotImplementedError(f'fname={fname}')
        elif fname == cls.ALL_SUBTREE:
            raise NotImplementedError(f'fname={fname}')
        elif fname == cls.LEAVES_ALL:
            raise NotImplementedError(f'fname={fname}')
        else:
            raise ValueError(f'{fname} invalid')

    @classmethod
    def acquire_build_mask_function(cls, fname, mutual_level=5):
        if fname == cls.DEFAULT:
            return cls.build_mask_default
        elif fname == cls.ALL_ALL:
            return cls.build_mask_default
        elif fname == cls.LEAVES_SUBTREE:
            return cls.build_mask_leaves_subtree
        else:
            raise ValueError(f'{fname} invalid')

    @classmethod
    def build_mask_default(cls, num_heads, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        att_nelems = nsent * (tk + nk)
        if key_pad is not None:
            assert node_pad is not None
            pad_mask = torch.cat([key_pad, node_pad], 2).view(bsz, 1, 1, att_nelems).expand(bsz, num_heads, 1, att_nelems)
            pad_mask = pad_mask.contiguous().view(bsz * num_heads, 1, att_nelems)
        else:
            pad_mask = None
        return pad_mask

    @classmethod
    def build_mask_all_all(cls, num_heads, device, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        return cls.build_mask_default(num_heads, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs)

    @classmethod
    def build_mask_leaves_subtree(cls, num_heads, device, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        leave_pad_mask = cls.build_mask_leaves_only(
            num_heads, device, key_pad, node_pad, node_indices, bsz, tk, nk, nsent
        )
        node_pad_mask = cls.build_mask_subtree_only(
            num_heads, key_pad, node_pad, node_indices, bsz, nk, tk, nk, nsent
        )
        # leave_pad_mask:       [b, 1,  1, 1, m * (tk + nk)]
        # node_pad_mask:        [b, h, nk, m, m * (tk + nk)]
        return (leave_pad_mask, node_pad_mask)

    @classmethod
    def mask_weights_wmask(cls, attn_weights, pad_mask):
        # attn_weights:     [b * h, tq, mtnk]
        # pad_mask:         [b * h, *, mtnk]
        # todo: try using inplace ops
        if pad_mask is None:
            return attn_weights
        attn_weights = attn_weights.float().masked_fill_(pad_mask, float('-inf')).type_as(attn_weights)
        return attn_weights

    @classmethod
    def mask_weights_leave_subtree_wmask(cls, attn_weights, pad_mask):
        assert isinstance(pad_mask, tuple)
        leave_pad_mask, node_pad = pad_mask
        b, h, nk, m, mtknk = node_pad.size()
        tknk = mtknk // m
        tk = tknk - nk
        all_weights = attn_weights.view(b, h, tknk, m, mtknk).float()
        all_weights[:, :, :tk].masked_fill_(leave_pad_mask, float('-inf'))
        all_weights[:, :, tk:].masked_fill_(node_pad, float('-inf'))
        all_weights = all_weights.type_as(attn_weights)
        all_weights = all_weights.view(b * h, (tk + nk) * m, m * (tk + nk))
        return all_weights

    @classmethod
    def wmask_default(cls, self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        att_nelems = nsent * (tk + nk)
        if key_pad is not None:
            assert node_pad is not None
            attn_weights = attn_weights.view(bsz, self.num_heads, tq, att_nelems)
            pad_mask = torch.cat([key_pad, node_pad], 2).view(bsz, 1, 1, att_nelems)
            assert not self.onnx_trace
            # src_lens_denom = (~pad_mask).type_as(attn_weights).sum(dim=-1, keepdim=True).clamp_(min=1.0, max=10000)
            # attn_weights = self.maybe_norm_src_len(attn_weights, src_lens_denom)
            attn_weights = attn_weights.float().masked_fill(pad_mask, float('-inf')).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, tq, att_nelems)

        return attn_weights

    @classmethod
    def wmask_leaves_subtree(cls, self, att_w, key_pad, node_pad, node_idx, bsz, tq, tk, nk, nsent, **kwargs):
        # fixme: leaves on leaves, nodes on subtree (include leaves)

        leave_w, node_w = cls.split_attn_weights(self, att_w, bsz, tq, tk, nk, nsent)
        leave_w = cls.mask_leaves_only(self, leave_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        node_w = cls.mask_subtree_only(self, node_w, key_pad, node_pad, node_idx, bsz, tk, nk, nsent)
        att_w = cls.merge_attn_weights(self, leave_w, node_w, bsz, tq, tk, nk, nsent)
        return att_w

    @classmethod
    def build_mask_leaves_only(cls, num_heads, device, key_pad, node_pad, node_indices, bsz, tk, nk, m, **kwargs):
        # weights:                  [b, h, tk + nk, m, m, tk + nk]
        # rs_weights:               [b, h, tk + nk, m, m * (tk + nk)]
        # key_pad:                  [b, m, tk]
        # node_pad:                 [b, m, nk]
        # pad_mask:                 [b, 1,       1, 1, mtknk]
        att_nelems = m * (tk + nk)
        if node_pad is not None:
            node_pad = torch.ones_like(node_pad, dtype=node_pad.dtype, device=node_pad.device)
        else:
            assert key_pad is None
            node_pad = torch.ones(bsz, m, nk, dtype=torch.uint8, device=device)
            key_pad = torch.zeros(bsz, m, tk, dtype=torch.uint8, device=device)
        pad_mask = torch.cat([key_pad, node_pad], 2)
        pad_mask = pad_mask.view(bsz, 1, 1, 1, att_nelems)
        return pad_mask

    @classmethod
    def build_mask_subtree_only(cls, num_heads, key_pad, node_pad, node_indices, bsz, tq, tk, nk, m, **kwargs):
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
            # node_leaves = node_leaves.type_as(weights)
            node_leaves = node_leaves.float()
            node_npad = torch.tril(torch.matmul(node_leaves, node_leaves.transpose(3, 4)).clamp_(0, 1)).type_as(leave_mask)
            # node_mask:                [b, h, m, nk, nk]
            node_npad = node_npad.view(bsz, num_heads, m, tq, 1, tq).transpose(2, 3)
            node_mask = (~node_npad) | node_pad

            # leave_mask:                   [b, h, t, m, m, tk]
            # node_mask:                    [b, h, t, m, m, nk]
            # pad_mas:                      [b, h, t, m, m * (tk + nk)]
            pad_mask = torch.cat([leave_mask, node_mask], 5)
            pb, ph, pt, pm, pm_, pntk = pad_mask.size()
            pad_mask = pad_mask.view(pb, ph, pt, pm, pm_ * pntk)
        return pad_mask




# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.ln = torch.nn.Linear(20, 5)
#         self.ln2 = torch.nn.Linear(5, 1)
#     def forward(self, x, m):
#         o1 = self.ln(x)
#         o1[:, :2].masked_fill_(m, 0)
#         o3 = self.ln2(o1)
#         print(o1)
#         return o3.sum()
# # test
# x = torch.Tensor(5, 20).uniform_().cuda()
# m = torch.ones(5, 2).byte().cuda()
# model = Model()
# model.cuda()
# optimizer = torch.optim.SGD(model.parameters(), 0.1)
# optimizer.zero_grad()
# loss = model(x, m)
# loss.backward()
# optimizer.step()


class EffDistinctStackNodesOnAffinityValueAttention(nn.Module):
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
        self.nstack_mask_func = WeightMaskEff.acquire_wmask_with_mask_function(self.nstack_mask_fname, self.mutual_level)
        self.build_mask_fn = WeightMaskEff.acquire_build_mask_function(self.nstack_mask_fname, self.mutual_level)

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
        return 'sln={},on={},posemb={},posemb_l={},hier_emb={},cumnode={},linear={},wfname={},upnorm={},hier_share={},dwstack_proj_act={}'.format(
            self.src_len_norm, self.nstack_on, self.nstack_pos_embed, self.nstack_pos_embed_learned,
            self.nstack_hier_embed, self.cum_node,
            self.nstack_linear, self.nstack_mask_fname, self.wnstack_up_norm, self.nstack_hier_embed_share, self.dwstack_proj_act
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
            new_embed = nstack_att.HierarchicalEmbedding(
                self.args, self.head_dim, self.num_heads,
                self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver)
            if self.nstack_hier_embed_share:
                print(f'Set embed to args')
                self.args.ShareHierarchicalEmbedding = new_embed
            return new_embed

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
            # fl_mask = mask.long().squeeze(-1).transpose(2, 3).contiguous().view(b * m * n, t)
            # fl_mask += self.embed_positions.padding_idx
            #
            # embeds = self.embed_positions(fl_mask)
            # emb_dim = embeds.size(-1)
            # embeds = embeds.view(b, m, n, t, emb_dim).transpose(2, 3)
            # return embeds
            raise NotImplementedError
        elif self.hier_embed_positions is not None:
            embeds = self.hier_embed_positions(mask)
            # return embeds
            raise NotImplementedError
        else:
            return 0.0

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
            node_states /= torch.cumsum(float_mask, dim=3).clamp_(1.0, 1e4)
        elif self.wnstack_up_norm == 'sqrt_mean':
            node_states /= torch.cumsum(float_mask, dim=3).sqrt_().clamp_(1.0, 1e4)
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

    @classmethod
    def build_nstack_mask(cls, tk, device, indices):
        """

        :param tk:
        :param device:
        :param indices:      [b, m, n, 2]
        :return:
        """
        indices = indices.unsqueeze(2)
        arange = torch.arange(0, tk, device=device, dtype=indices.dtype).view(1, 1, tk, 1, 1)
        left_mask = arange >= indices[:, :, :, :, :1]
        right_mask = arange > indices[:, :, :, :, 1:]
        mask = left_mask ^ right_mask
        return mask

    def _compute_nstree_states(self, leaves, rv_nodes, mask, hier_embed=None, is_affinity=True, **kwargs):
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
        # b__, m__, n_, _ = indices.size()
        assert b == b_, f'{b} != {b_}'
        assert m == m_, f'{m} != {m_}'
        assert c == c_, f'{c} != {c_}'
        # assert n_ == n, f'{n} != {n_}'
        device = leaves.device
        leaves = leaves.unsqueeze(3)
        # indices = indices.unsqueeze(2)
        rv_nodes = rv_nodes.unsqueeze(2)
        # leaves:                   [b, m, t, 1, c]
        # nodes:                    [b, m, 1, n, c]
        # indices:                  [b, m, 1, n, 2]
        # arange:                   [1, 1, t, 1, 1]
        # mask:                     [b, m, t, n, 1]
        # pos_embed:                [b, m, t, n, c]
        # more to outside
        # arange = torch.arange(0, t, device=device, dtype=indices.dtype).view(1, 1, t, 1, 1)
        # left_mask = arange >= indices[:, :, :, :, :1]
        # right_mask = arange > indices[:, :, :, :, 1:]
        # mask = left_mask ^ right_mask

        # get_pos_embed = kwargs.get('get_pos_embed', True)
        # if is_affinity:
        #     nodes = rv_nodes
        # else:
        #     pos_embed = self.get_pos_embed(mask)
        #     nodes = rv_nodes + pos_embed

        nodes = rv_nodes
        if hier_embed is not None:
            nodes = rv_nodes + hier_embed

        assert not torch.isnan(rv_nodes).any()

        node_states = self.accumulate_upward(leaves, nodes, mask, is_affinity=is_affinity, **kwargs)

        assert not torch.isnan(node_states).any()
        assert not torch.isinf(node_states).any()
        rv_node_out = self.accumulate_rightward(node_states, mask, leaves, **kwargs)
        assert not torch.isnan(rv_node_out).any()

        return rv_node_out

    def compute_att_weights(self, q, k, v, node_k, node_v, nstack_mask, key_pad, node_pad, **kwargs):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param nstack_mask:     [b * h, m, tk, nk, 1]
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

        attn_leaves = torch.bmm(q, k.view(bh, nsent * tk, d).transpose(1, 2)).view(bh, tq, nsent, tk).permute(
            0, 2, 3, 1)
        attn_nodes = torch.bmm(q, node_k.view(bh, nsent * nk, d).transpose(1, 2)).view(bh, tq, nsent, nk).permute(
            0, 2, 3, 1)
        # att_leaves:           [b * h, m, tk, tq]
        # att_nodes:            [b * h, m, nk, tq]

        nstree_attn_nodes = self._compute_nstree_states(
            attn_leaves, attn_nodes, nstack_mask, is_affinity=True, **kwargs)

        nstree_attn_weights = torch.cat([attn_leaves, nstree_attn_nodes], 2).view(bh, att_nelems, tq).transpose(1, 2)

        node_v = self._compute_nstree_states(v, node_v, nstack_mask, is_affinity=False, **kwargs)
        values = torch.cat([v, node_v], 2).view(bh, att_nelems, d)
        # else:

        return nstree_attn_weights, values

    def mask_weights(self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs):
        return self.nstack_mask_func(
            self, attn_weights, key_pad, node_pad, node_indices, bsz, tq, tk, nk, nsent, **kwargs
        )

    def mask_weights_with_mask(self, attn_weights, pad_mask):
        return self.nstack_mask_func(attn_weights, pad_mask)

    def compute_nstack_att(
            self, q, k, v, node_k, node_v, nstack_mask, key_pad, node_pad, pad_mask, src_len, tgt_len, node_src_len,
            bsz, need_weights, **kwargs
    ):
        """

        :param q:               [b * h, tq, d]
        :param k:               [b * h, m, tk, d]
        :param v:               [b * h, m, tk, d]
        :param node_k:          [b * h, m, nk, d]
        :param node_v:          [b * h, m, nk, d]
        :param nstack_mask:     [b * h, m, tk, nk, 1]
        :param key_pad:         [b, m, tk]
        :param node_pad:        [b, m, nk]
        :param pad_mask:        built outside
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
            q, k, v, node_k, node_v, nstack_mask, key_pad, node_pad, **kwargs
        )

        # att_weights:          [b * h, tq, m * (tk + nk)]

        assert not torch.isnan(attn_weights).any(), f'attn_weights: before mask'
        # attn_weights = self.mask_weights(attn_weights, key_pad, node_pad, nstack_mask, bsz, tq, tk, nk, nsent)
        attn_weights = self.mask_weights_with_mask(attn_weights, pad_mask=pad_mask)
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

    def forward(self, query, key, value,
            node_key, node_value,
            nstack_mask,
            prebuilt_pad_mask,
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
        :param nstack_mask:         [b * h, m, tk, nk, 1]
        :param prebuilt_pad_mask:   built outside
        :param key_padding_mask:    [b, m, tk]
        :param node_padding_mask:   [b, m, nk]
        :param incremental_state:
        :param compute_query_nodes:
        :param compute_key_nodes:
        :param need_weights:
        :param static_kv:
        :param attn_mask:
        :param force_self_att:
        :return:
        """
        tq, bsz, dim = query.size()
        tk, bsz_k, m, dim_k = key.size()
        nk, bsz_k_, nsent_, dim_k_ = node_key.size()
        # nk_, bsz_i, nsent__, idim = indices.size()

        assert bsz == bsz_k
        assert bsz_k_ == bsz_k
        # assert nk == nk_, f'{node_key.size()} != {indices.size()}'
        # assert idim == 2
        h = self.num_heads
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
            'ori_prebuilt_pad_mask': prebuilt_pad_mask
        }

        f_key = key.view(tk, bsz_k * m, dim_k)
        f_value = value.view(tk, bsz_k * m, dim_k)

        f_node_key = node_key.view(nk, bsz_k * m, dim_k)
        f_node_value = node_value.view(nk, bsz_k * m, dim_k)

        # f_key_pad_mask = key_padding_mask = key_padding_mask.view(key_bsz * nsent, tk)
        assert not torch.isnan(query).any()

        (q, f_key, f_value, f_node_key, f_node_value, key_padding_mask, node_padding_mask,
         saved_state, src_len, node_src_len, tgt_len, query_bsz) = self.prepare_dptree_qkv(
            query, f_key, f_value, f_node_key, f_node_value,
            m,
            key_padding_mask, node_padding_mask,
            incremental_state, need_weights, static_kv,
            compute_query_nodes=compute_query_nodes, compute_key_nodes=compute_key_nodes, force_self_att=force_self_att
        )
        # q:    [bq * h, tq, d]
        # fk:   [bk * m * h, tk, d]
        # fv:   [bk * m * h, tk, d]
        # fpad: [bk, m, tk]
        if key_padding_mask is not None:
            assert not torch.isnan(key_padding_mask).any()

        f_key = f_key.view(bsz_k, m, h, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_key = f_key.view(bsz_k * h, m, tk, self.head_dim)

        f_value = f_value.view(bsz_k, m, h, tk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_value = f_value.view(bsz_k * h, m, tk, self.head_dim)

        f_node_key = f_node_key.view(bsz_k, m, h, nk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_node_key = f_node_key.view(bsz_k * h, m, nk, self.head_dim)

        f_node_value = f_node_value.view(bsz_k, m, h, nk, self.head_dim).permute(0, 2, 1, 3, 4).contiguous()
        f_node_value = f_node_value.view(bsz_k * h, m, nk, self.head_dim)

        kwargs['after_query'] = q

        (attn, attn_weights) = self.compute_nstack_att(
            q, f_key, f_value, f_node_key, f_node_value, nstack_mask, key_padding_mask, node_padding_mask,
            prebuilt_pad_mask, src_len, tgt_len, node_src_len, bsz, need_weights
        )
        return attn, attn_weights









