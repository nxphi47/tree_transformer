import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
# itertools.chain(*my_list_of_lists)

import os
from fairseq import options
from fairseq import utils

from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LearnedPositionalEmbedding, MultiheadAttention,
    SinusoidalPositionalEmbedding
)

from .dptree_multihead_attention import DPTreeMultiheadAttention
from .dptree_sep_multihead_attention import DPTreeSeparateOnlyKeyAttention

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model,
    register_model_architecture,
)

from ..modules.embeddings import *

from fairseq.models.transformer import *
from .dptree_transformer_layer import *
# from ..modules.nstack_tree_attention import *
# from ..modules.nstack_tree_attention import *
from .nstack_tree_attention import *
from .nstack_merge_tree_attention import *
DEBUG = False


def print_debug(s):
    if DEBUG:
        print('nstack_transformer_layer.py::' + s)


class NstackTransformerEncoderLayer(nn.Module):
    def __init__(self, args, padding_idx=1, compute_nodes=True):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class
        self.padding_idx = padding_idx
        self.compute_nodes = compute_nodes

        # self.self_attn = DPTreeMultiheadAttention(
        self.self_attn = self.dptree_class(
            args, self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
            padding_idx=self.padding_idx
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = getattr(args, 'input_dropout', 0)

        self.relu_dropoute_layer = nn.Dropout(self.relu_dropout)
        self.plain_dropoute_layer = nn.Dropout(self.dropout)
        self.input_dropout_layer = nn.Dropout(self.input_dropout)

        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def extra_repr(self):
        return f'compute_nodes={self.compute_nodes}'

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    @property
    def recusive_params(self):
        try:
            params = self.self_attn.recusive_params
        except Exception as e:
            params = []
        return params

    def forward(self, leave_x, node_x, indices, encoder_padding_mask, node_padding_mask):
        """

        :param leave_x:                 [tq, b, m, C]
        :param node_x:                  [nq, b, m, C]
        :param indices:                 [nq, b, m, 2]
        :param encoder_padding_mask:    [b, m, tq]
        :param node_padding_mask:       [b, m, nq]
        :return: encoded output shape
        """
        # residual = x
        assert not torch.isnan(leave_x).any()
        tq, bsz, nsent, dim = leave_x.size()
        nq, _, _, _ = node_x.size()
        nq_, _, _, _ = indices.size()
        assert nq == nq_, f'{nq} != {nq_}'

        leaves = leave_x
        nodes = node_x

        x = torch.cat([leaves, nodes], 0)
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_layer(x)
        # tq, bsz, nsent, dim = x.size()
        query = x.transpose(1, 2).contiguous().view((tq + nq) * nsent, bsz, dim)
        k_leaves = x[:tq]
        k_nodes = x[tq:]
        assert not torch.isnan(x).any(), f'x problem'
        assert not torch.isnan(query).any(), f'query problem'

        # TODO--here
        x, weights = self.self_attn(
            query=query,
            key=k_leaves, value=k_leaves,
            node_key=k_nodes, node_value=k_nodes,
            indices=indices,
            key_padding_mask=encoder_padding_mask,
            node_padding_mask=node_padding_mask,
            force_self_att=True,
            compute_query_nodes=self.compute_nodes,
            compute_key_nodes=self.compute_nodes,
        )
        assert not torch.isnan(x).any(), f'after self attention problem'

        # tq, nsent, new_dim = x.size()
        x = x.contiguous().view((tq + nq), nsent, bsz, dim).transpose(1, 2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x

        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        o_leaves = x[:tq]
        o_nodes = x[tq:]

        return o_leaves, o_nodes, weights


class NstackEffTransformerEncoderLayer(NstackTransformerEncoderLayer):

    def build_nstack_mask(self, tk, device, indices):
        return self.self_attn.build_nstack_mask(tk, device, indices)

    def build_pad_mask(self, key_pad, node_pad, bsz, tq, tk, nk, nsent, device, indices):
        # bsz =
        pad_mask = self.build_mask_fn(
            num_heads=self.self_attn.num_heads,
            device=device, key_pad=key_pad, node_pad=node_pad, node_indices=indices,
            bsz=bsz, tq=tq, tk=tk, nk=nk, nsent=nsent
        )
        return pad_mask

    def forward(self, leave_x, node_x, nstack_mask, encoder_padding_mask, node_padding_mask, prebuilt_pad_mask):
        """

        :param leave_x:                 [tq, b, m, C]
        :param node_x:                  [nq, b, m, C]
        :param nstack_mask:             [b * h, m, tk, nk, 1]
        :param encoder_padding_mask:    [b, m, tq]
        :param node_padding_mask:       [b, m, nq]
        :param prebuilt_pad_mask:       built outside
        :return: encoded output shape
        """
        # residual = x
        assert not torch.isnan(leave_x).any()
        tq, bsz, nsent, dim = leave_x.size()
        nq, _, _, _ = node_x.size()
        # nq_, _, _, _ = indices.size()
        # assert nq == nq_, f'{nq} != {nq_}'

        leaves = leave_x
        nodes = node_x

        x = torch.cat([leaves, nodes], 0)
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)

        # tq, bsz, nsent, dim = x.size()
        query = x.transpose(1, 2).contiguous().view((tq + nq) * nsent, bsz, dim)
        k_leaves = x[:tq]
        k_nodes = x[tq:]
        assert not torch.isnan(x).any(), f'x problem'
        assert not torch.isnan(query).any(), f'query problem'

        # TODO--here
        x, weights = self.self_attn(
            query=query,
            key=k_leaves, value=k_leaves,
            node_key=k_nodes, node_value=k_nodes,
            nstack_mask=nstack_mask, prebuilt_pad_mask=prebuilt_pad_mask,
            key_padding_mask=encoder_padding_mask,
            node_padding_mask=node_padding_mask,
            force_self_att=True,
            compute_query_nodes=self.compute_nodes,
            compute_key_nodes=self.compute_nodes,
        )
        assert not torch.isnan(x).any(), f'after self attention problem'

        # tq, nsent, new_dim = x.size()
        x = x.contiguous().view((tq + nq), nsent, bsz, dim).transpose(1, 2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x

        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        o_leaves = x[:tq]
        o_nodes = x[tq:]

        return o_leaves, o_nodes, weights


class Nstack2SeqTransformerDecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before
        self.dptree_class = args.dptree_class
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.nstack_cross = getattr(args, 'nstack_cross', True)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
            cross_kwargs = {
                'nstack_mask_fn': getattr(args, 'cross_nstack_mask_fn', args.nstack_mask_fn)
            }
            if self.nstack_cross:
                print(f'Build Cross attention: {self.dptree_class}')
                self.encoder_attn = self.dptree_class(
                    args, self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout, **cross_kwargs
                )
            else:
                self.encoder_attn = MultiheadAttention(
                    self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def forward(
            self,
            x,
            encoder_leaves,
            encoder_nodes,
            encoder_indices,
            encoder_padding_mask,
            node_padding_mask,

            incremental_state,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None):
        # encoder_leaves = encoder_leaves,
        # encoder_nodes = encoder_nodes,
        # encoder_indices = encoder_indices,
        # encoder_padding_mask = encoder_padding_mask,
        # node_padding_mask = node_padding_mask,

        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None

        # TODO: attention here!
        need_weights = (not self.training and self.need_attn)
        print(f'Cross att: need_weights={need_weights}')
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            if self.nstack_cross:
                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_leaves, value=encoder_leaves,
                    node_key=encoder_nodes, node_value=encoder_nodes,
                    indices=encoder_indices,
                    key_padding_mask=encoder_padding_mask,
                    node_padding_mask=node_padding_mask,

                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_weights,
                )
            else:
                # b, m, t, c = encoder_leaves.size()
                t, b, m, c = encoder_leaves.size()
                encoder_leaves_fl = encoder_leaves.permute(2, 0, 1, 3).contiguous().view(m * t, b, c)
                pad_mask = encoder_padding_mask.view(b, m * t) if encoder_padding_mask is not None else None

                x, attn = self.encoder_attn(
                    query=x,
                    key=encoder_leaves_fl,
                    value=encoder_leaves_fl,
                    key_padding_mask=pad_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_weights,
                )

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn


class NstackTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.args = args
        self.embed_dropout_layer = nn.Dropout(self.dropout)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        # TODO: what is this?
        self.max_source_positions = args.max_source_positions
        # TODO: set params
        # assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        # assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        assert not left_pad
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_path = args.encoder_embed_path

        self.embed_path_exists = self.embed_path is not None and os.path.exists(self.embed_path)
        self.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)
        self.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)

        self.vanilla_layers = getattr(args, 'vanilla_layers', 0)

        self.attention_rerun = getattr(args, 'attention_rerun', 1)
        assert self.attention_rerun >= 1
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.leave_embed_scale = 1.0 if self.embed_path_exists and self.embed_pretrained_no_scale else self.embed_scale
        self.node_embed_scale = self.embed_scale
        print(f'leave_embed_scale={self.leave_embed_scale}, node_embed_scale={self.node_embed_scale}')

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.node_embed_init = getattr(args, 'node_embed_init', 'embed')

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NstackTransformerEncoderLayer(
                args, padding_idx=self.padding_idx,
                compute_nodes=(i > 0) or not self.first_layer_nonodes)
            for i in range(args.encoder_layers)
        ])

        if self.vanilla_layers > 0:
            self.vanilla_leave_layers = nn.ModuleList([])
            self.vanilla_leave_layers.extend([
                TransformerEncoderLayer(args)
                for i in range(self.vanilla_layers)
            ])
        else:
            self.vanilla_leave_layers = None

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def extra_repr(self):
        return f'rerun={self.attention_rerun}'

    def embed(self, flat_src_tokens, **kwargs):
        assert not isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding)
        embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def embed_nodes(self, leave_embed, flat_node_tokens, **kwargs):
        if self.node_embed_init == 'embed':
            if self.args.pretrain_embed_mode == 'bert':
                return self.embed_tokens(flat_node_tokens, only_embedding=True)
            return self.embed(flat_node_tokens)
        elif self.node_embed_init == 'zero':
            b, l = flat_node_tokens.size()
            return torch.zeros(b, l, leave_embed.size(-1), dtype=leave_embed.dtype, device=leave_embed.device)
        else:
            raise ValueError(f'{self.node_embed_init} ???')

    @property
    def recusive_params(self):
        params = list(itertools.chain(*[x.recusive_params for x in self.layers]))
        return params

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs):
        """

        :param src_node_leaves:     [b, m, t]
        :param src_node_nodes:      [b, m, n]
        :param src_node_indices:    [b, m, n, 2]
        :param src_sent_lengths:    [b, m]
        :param kwargs:
        :return:
        """
        assert (src_node_leaves < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_leaves.max()}'
        assert (src_node_nodes < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_nodes.max()}'

        bsz, nsent, leave_len = src_node_leaves.size()
        bsz, nsent, node_len = src_node_nodes.size()

        flat_src_node_leaves = src_node_leaves.view(bsz, nsent * leave_len)
        flat_src_node_nodes = src_node_nodes.view(bsz, nsent * node_len)
        # flat_indices = src_indices.view(bsz * nsent, src_len, 2)

        leave_embeddings = self.embed(flat_src_node_leaves)
        node_embeddings = self.embed_nodes(leave_embeddings, flat_src_node_nodes)
        # print(f'embedding-weight::{self.embed_tokens.weight.max()}-n{self.embed_tokens.weight.max()}==={self.embed_tokens.weight.min()}-n{self.embed_tokens.weight.min()}')
        # print(f'forward-before-scale::{leave_embeddings.max()}-n{node_embeddings.max()}==={leave_embeddings.min()}-n{node_embeddings.min()}')

        leave_x = self.leave_embed_scale * leave_embeddings
        node_x = self.node_embed_scale * node_embeddings
        # print(f'forward-before-embed_positions::{leave_x.abs().max()}-n{node_x.abs().max()}')

        if self.embed_positions is not None:
            leave_x += self.embed_positions(flat_src_node_leaves)
        # leave_x = F.dropout(leave_x, p=self.dropout, training=self.training)
        # node_x = F.dropout(node_x, p=self.dropout, training=self.training)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.view(bsz, nsent, leave_len, leave_x.size(-1)).permute(2, 0, 1, 3)
        node_x = node_x.view(bsz, nsent, node_len, node_x.size(-1)).permute(2, 0, 1, 3)
        indices = src_node_indices.permute(2, 0, 1, 3)
        # :param
        # indices: [nq, b, m, 2]
        # [tq, b, m, C]
        # [nq, b, m, C]

        key_padding_mask = src_node_leaves.eq(self.padding_idx)
        node_padding_mask = src_node_nodes.eq(self.padding_idx)
        if not key_padding_mask.any() and not node_padding_mask.any():
            key_padding_mask = node_padding_mask = None

        if self.vanilla_layers > 0:
            # leaves encoder layers
            leave = leave_x.permute(2, 0, 1, 3).view(nsent * leave_len, bsz, leave_x.size(-1))
            encoder_padding_mask = src_node_leaves.eq(self.padding_idx).view(bsz, nsent * leave_len)
            # leave: [m, tq, b, c]
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, encoder_padding_mask)

            leave_x = leave.contiguous().view(nsent, leave_len, bsz, leave.size(-1)).permute(1, 2, 0, 3)
            # [tq, b, m, C]

        attention_dict = {}
        for j in range(self.attention_rerun):
            for i, layer in enumerate(self.layers):
                leave_x, node_x, weights = layer(leave_x, node_x, indices, key_padding_mask, node_padding_mask)
                attention_dict[f'att_{i}'] = weights

        x = torch.cat([leave_x, node_x], 0)
        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim

        out_dict = {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_node_indices,  # B x m x T x 2
            'encoder_padding_mask': key_padding_mask,  # B x m x T
            'node_padding_mask': node_padding_mask,  # B x m x T
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['node_padding_mask'] is not None:
            encoder_out['node_padding_mask'] = encoder_out['node_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)
        for k in encoder_out.keys():
            if "att_" in k:
                encoder_out[k] = encoder_out[k].index_select(0, new_order)

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class NstackEffTransformerEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.args = args
        self.embed_dropout_layer = nn.Dropout(self.dropout)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        # TODO: what is this?
        self.max_source_positions = args.max_source_positions
        # TODO: set params
        # assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        # assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        assert not left_pad
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_path = args.encoder_embed_path

        self.embed_path_exists = self.embed_path is not None and os.path.exists(self.embed_path)
        self.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)
        self.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)

        self.vanilla_layers = getattr(args, 'vanilla_layers', 0)

        self.attention_rerun = getattr(args, 'attention_rerun', 1)
        assert self.attention_rerun >= 1
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)

        self.leave_embed_scale = 1.0 if self.embed_path_exists and self.embed_pretrained_no_scale else self.embed_scale
        self.node_embed_scale = self.embed_scale
        print(f'leave_embed_scale={self.leave_embed_scale}, node_embed_scale={self.node_embed_scale}')

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.node_embed_init = getattr(args, 'node_embed_init', 'embed')

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NstackEffTransformerEncoderLayer(
                args, padding_idx=self.padding_idx,
                compute_nodes=(i > 0) or not self.first_layer_nonodes)
            for i in range(args.encoder_layers)
        ])

        if self.vanilla_layers > 0:
            self.vanilla_leave_layers = nn.ModuleList([])
            self.vanilla_leave_layers.extend([
                TransformerEncoderLayer(args)
                for i in range(self.vanilla_layers)
            ])
        else:
            self.vanilla_leave_layers = None

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def extra_repr(self):
        return f'rerun={self.attention_rerun}'

    def embed(self, flat_src_tokens, **kwargs):
        assert not isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding)
        embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def embed_nodes(self, leave_embed, flat_node_tokens, **kwargs):
        if self.node_embed_init == 'embed':
            if self.args.pretrain_embed_mode == 'bert':
                return self.embed_tokens(flat_node_tokens, only_embedding=True)
            return self.embed(flat_node_tokens)
        elif self.node_embed_init == 'zero':
            b, l = flat_node_tokens.size()
            return torch.zeros(b, l, leave_embed.size(-1), dtype=leave_embed.dtype, device=leave_embed.device)
        else:
            raise ValueError(f'{self.node_embed_init} ???')

    @property
    def recusive_params(self):
        params = list(itertools.chain(*[x.recusive_params for x in self.layers]))
        return params

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs):
        """

        :param src_node_leaves:     [b, m, t]
        :param src_node_nodes:      [b, m, n]
        :param src_node_indices:    [b, m, n, 2]
        :param src_sent_lengths:    [b, m]
        :param kwargs:
        :return:
        """
        assert (
                    src_node_leaves < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_leaves.max()}'
        assert (src_node_nodes < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_nodes.max()}'

        bsz, nsent, leave_len = src_node_leaves.size()
        bsz, nsent, node_len = src_node_nodes.size()

        flat_src_node_leaves = src_node_leaves.view(bsz, nsent * leave_len)
        flat_src_node_nodes = src_node_nodes.view(bsz, nsent * node_len)
        # flat_indices = src_indices.view(bsz * nsent, src_len, 2)

        leave_embeddings = self.embed(flat_src_node_leaves)
        node_embeddings = self.embed_nodes(leave_embeddings, flat_src_node_nodes)
        # print(f'embedding-weight::{self.embed_tokens.weight.max()}-n{self.embed_tokens.weight.max()}==={self.embed_tokens.weight.min()}-n{self.embed_tokens.weight.min()}')
        # print(f'forward-before-scale::{leave_embeddings.max()}-n{node_embeddings.max()}==={leave_embeddings.min()}-n{node_embeddings.min()}')

        leave_x = self.leave_embed_scale * leave_embeddings
        node_x = self.node_embed_scale * node_embeddings
        # print(f'forward-before-embed_positions::{leave_x.abs().max()}-n{node_x.abs().max()}')

        if self.embed_positions is not None:
            leave_x += self.embed_positions(flat_src_node_leaves)
        # leave_x = F.dropout(leave_x, p=self.dropout, training=self.training)
        # node_x = F.dropout(node_x, p=self.dropout, training=self.training)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.view(bsz, nsent, leave_len, leave_x.size(-1)).permute(2, 0, 1, 3)
        node_x = node_x.view(bsz, nsent, node_len, node_x.size(-1)).permute(2, 0, 1, 3)
        indices = src_node_indices.permute(2, 0, 1, 3)
        device = leave_x.device
        # :param
        # indices: [nq, b, m, 2]
        # [tq, b, m, C]
        # [nq, b, m, C]

        key_padding_mask = src_node_leaves.eq(self.padding_idx)
        node_padding_mask = src_node_nodes.eq(self.padding_idx)
        if not key_padding_mask.any() and not node_padding_mask.any():
            key_padding_mask = node_padding_mask = None

        # todo: pre-building masks
        """
        def build_nstack_mask(self, tk, device, indices):
            return self.self_attn.build_nstack_mask(tk, device, indices)

        def build_pad_mask(self, key_pad, node_pad, bsz, tq, tk, nk, nsent, device, indices):
            # bsz =
            pad_mask = self.build_mask_fn(
                num_heads=self.self_attn.num_heads,
                device=device, key_pad=key_pad, node_pad=node_pad, node_indices=indices,
                bsz=bsz, tq=tq, tk=tk, nk=nk, nsent=nsent
            )
            return pad_mask
        """
        tq = nsent * (leave_len + node_len)
        nstack_mask = self.layers[0].build_nstack_mask(leave_len, device, src_node_indices)
        prebuilt_pad_mask = self.layers[0].build_pad_mask(
            key_padding_mask, node_padding_mask, bsz, tq, leave_len, node_len, nsent, device, src_node_indices
        )

        if self.vanilla_layers > 0:
            # leaves encoder layers
            leave = leave_x.permute(2, 0, 1, 3).view(nsent * leave_len, bsz, leave_x.size(-1))
            encoder_padding_mask = src_node_leaves.eq(self.padding_idx).view(bsz, nsent * leave_len)
            # leave: [m, tq, b, c]
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, encoder_padding_mask)

            leave_x = leave.contiguous().view(nsent, leave_len, bsz, leave.size(-1)).permute(1, 2, 0, 3)
            # [tq, b, m, C]

        attention_dict = {}
        for j in range(self.attention_rerun):
            for i, layer in enumerate(self.layers):
                leave_x, node_x, weights = layer(
                    leave_x, node_x, nstack_mask, key_padding_mask, node_padding_mask, prebuilt_pad_mask
                )
                # asdasd
                attention_dict[f'att_{i}'] = weights

        x = torch.cat([leave_x, node_x], 0)
        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim

        out_dict = {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_node_indices,  # B x m x T x 2
            'encoder_padding_mask': key_padding_mask,  # B x m x T
            'node_padding_mask': node_padding_mask,  # B x m x T
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['node_padding_mask'] is not None:
            encoder_out['node_padding_mask'] = encoder_out['node_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)
        for k in encoder_out.keys():
            if "att_" in k:
                encoder_out[k] = encoder_out[k].index_select(0, new_order)

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class Nstack2SeqTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_dropout_layer = nn.Dropout(self.dropout)

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        # TODO: set params
        assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            Nstack2SeqTransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.embed_dropout_layer(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # todo: retrieving cross attention
        # 'encoder_out': x,  # T x B x m x C
        # 'encoder_indices': src_node_indices,  # B x m x T x 2
        # 'encoder_padding_mask': key_padding_mask,  # B x m x T
        # 'node_padding_mask': node_padding_mask,  # B x m x T
        """
        :param key:                 [tk, b, m, c]
        :param value:               [tk, b, m, c]
        :param node_key:            [nk, b, m, c]
        :param node_value:          [nk, b, m, c]
        :param indices:             [nk, b, m, 2]
        """

        assert encoder_out is not None, f'encoder_out is None!'
        encoder_output = encoder_out['encoder_out']
        encoder_indices = encoder_out['encoder_indices'].permute(2, 0, 1, 3)
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        node_padding_mask = encoder_out['node_padding_mask']

        tnk, b_, m_, c = encoder_output.size()
        nk, b, m, _ = encoder_indices.size()
        tk = tnk - nk
        encoder_leaves = encoder_output[:tk]
        encoder_nodes = encoder_output[tk:]

        inner_atts = []

        for layer in self.layers:
            x, attn = layer(
                x=x,
                encoder_leaves=encoder_leaves,
                encoder_nodes=encoder_nodes,
                encoder_indices=encoder_indices,
                encoder_padding_mask=encoder_padding_mask,
                node_padding_mask=node_padding_mask,

                # encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
                # encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                # encoder_indices=encoder_out['encoder_indices'] if encoder_out is not None else None,
                # encoder_flat_indices=encoder_flat_indices,
                # encoder_gt_indices=encoder_gt_indices,
                incremental_state=incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
            inner_atts.append(attn)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states, 'inner_atts': inner_atts}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


# fixme: ------------------- nstack_merge family --------------------


class NstackMergeTransformerEncoderLayer(nn.Module):
    def __init__(self, args, padding_idx=1, compute_nodes=True, mask_default=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class
        self.padding_idx = padding_idx
        self.compute_nodes = compute_nodes
        self.mask_default = mask_default
        att_kwargs = {}
        # nstack_mask_fn
        if mask_default:
            att_kwargs['nstack_mask_fn'] = "default"

        self.self_attn = self.dptree_class(
            args, self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
            padding_idx=self.padding_idx, **att_kwargs
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = getattr(args, 'input_dropout', 0)

        self.relu_dropoute_layer = nn.Dropout(self.relu_dropout)
        self.plain_dropoute_layer = nn.Dropout(self.dropout)
        self.input_dropout_layer = nn.Dropout(self.input_dropout)

        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def extra_repr(self):
        return f'compute_nodes={self.compute_nodes},maskdf={self.mask_default}'

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    @property
    def recusive_params(self):
        try:
            params = self.self_attn.recusive_params
        except Exception as e:
            params = []
        return params

    def forward(self, x_le, x_no, ntree_mask, hier_embed, pad_mask, key_pad, node_pad):
        # x_le:             [n, b, c]
        # x_no:             [m, b, c]
        # ntree_mask:       [bh, n, m, 1]
        # hier_embed:       [bh, n, m, d] c=d*h
        # pad_mask:         [b, 1, n + m, n + m]
        n, b, dim = x_le.size()
        m, b_, dim_ = x_no.size()

        leaves = x_le
        nodes = x_no

        x = torch.cat((leaves, nodes), 0)
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x = self.input_dropout_layer(x)
        query = x
        k_le = x[:n]
        k_no = x[n:]

        x, weights = self.self_attn(
            query=query,
            key=k_le, value=k_le,
            node_key=k_no, node_value=k_no,
            ntree_mask=ntree_mask, hier_embed=hier_embed,
            pad_mask=pad_mask, key_pad=key_pad, node_pad=node_pad,
            force_self_att=True
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        # print(f'before fc1: {x.device}')
        # self.fc1.to(x.device)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x

        # assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        # assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        o_leaves = x[:n]
        o_nodes = x[n:]

        return o_leaves, o_nodes, weights

#
# class C(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.a = nn.Linear(10, 10)
#         self.layers = nn.ModuleList([])
#         self.layers.extend([
#             nn.Linear(5, 5) for i in range(5)
#         ])


class NstackMergeTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.args = args

        self.gpu_idx = None
        self.model_parallel = False

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        # TODO: what is this?
        self.max_source_positions = args.max_source_positions
        # TODO: set params
        # assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        # assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        assert not left_pad
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_path = args.encoder_embed_path

        self.embed_path_exists = self.embed_path is not None and os.path.exists(self.embed_path)
        self.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)
        self.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)
        self.pretrained_linear = getattr(args, 'pretrained_linear', False)
        self.use_pos = getattr(args, 'use_pos', False)

        self.vanilla_layers = getattr(args, 'vanilla_layers', 0)

        self.attention_rerun = getattr(args, 'attention_rerun', 1)
        assert self.attention_rerun >= 1
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_scale = math.sqrt(embed_dim)

        self.leave_embed_scale = 1.0 if self.embed_path_exists and self.embed_pretrained_no_scale else self.embed_scale
        self.node_embed_scale = self.embed_scale
        print(f'leave_embed_scale={self.leave_embed_scale}, node_embed_scale={self.node_embed_scale}')

        self.dptree_class = args.dptree_class

        self.node_embed_init = getattr(args, 'node_embed_init', 'embed')

        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

        self.nstack_mask_fname = getattr(args, 'nstack_mask_fn', 'default')
        self.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', None)
        if self.nstack_mask_df_layer is None:
            self.nstack_mask_df_layer = [False] * args.encoder_layers
        else:
            assert isinstance(self.nstack_mask_df_layer, (list, tuple))
            assert len(self.nstack_mask_df_layer) == args.encoder_layers, f'{len(self.nstack_mask_df_layer)}'
        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)
        self.nstack_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            self.nstack_mask_fname, self.mutual_level)
        self.nstack_default_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            'default', self.mutual_level)
        self.is_mask_default = self.nstack_mask_fname == 'default' or self.nstack_mask_fname == 'all_all'

        # ---------------- modules --------------------------
        self.embed_dropout_layer = nn.Dropout(self.dropout)
        self.embed_tokens = embed_tokens
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.hier_pos_positions = MergeHierarchicalEmbedding(
            args, args.encoder_layers, self.head_dim, self.heads,
            self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver,
            self.nstack_hier_embed_share
        ) if self.nstack_hier_embed else None

        self.pretrained_proj = Linear(
            self.encoder_embed_dim, self.encoder_embed_dim) if self.pretrained_linear else None

        if self.vanilla_layers > 0:
            self.vanilla_leave_layers = nn.ModuleList([])
            self.vanilla_leave_layers.extend([
                TransformerEncoderLayer(args)
                for i in range(self.vanilla_layers)
            ])
        else:
            self.vanilla_leave_layers = None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NstackMergeTransformerEncoderLayer(
                args, padding_idx=self.padding_idx,
                compute_nodes=(i > 0) or not self.first_layer_nonodes,
                mask_default=self.nstack_mask_df_layer[i]
            )
            for i in range(args.encoder_layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def setup_cuda(self, gpu_idx):
        print(f'[{self.__class__.__name__}] Setup gpu_idx: {gpu_idx}')
        self.gpu_idx = gpu_idx
        self.model_parallel = True

        first_gpu = [
            'embed_dropout_layer', 'embed_tokens', 'embed_positions', 'hier_pos_positions',
            'pretrained_proj', 'vanilla_leave_layers'
        ]
        last_gpu = ['layer_norm']
        for name, module in self.named_children():
            # module._apply(fn)
            if name in first_gpu:
                print(f'|| [0][{name}]: {module}')
                module.cuda(0)
            elif name in last_gpu:
                print(f'|| [{self.gpu_idx[-1]}][{name}]: {module}')
                module.cuda(self.gpu_idx[-1])
            else:
                assert name == 'layers'
                for i, layer in enumerate(self.layers):
                    print(f'|| [{self.gpu_idx[i]}][{name}]: {layer}')
                    layer.cuda(self.gpu_idx[i])

        for k, param in self._parameters.items():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.

                # param.data = fn(param.data)
                # if param._grad is not None:
                #     param._grad.data = fn(param._grad.data)
                raise NotImplementedError(f'param is not None [{k}]: {param}')

        for key, buf in self._buffers.items():
            if buf is not None:
                print(f'setup cuda for buf: {key}: {buf}')
                self._buffers[key] = buf.cuda(self.gpu_idx[-1])
                # raise NotImplementedError(f'buf is not None [{key}]: {buf}')
        return self

    # def cuda(self, device=None):
    #     return super().cuda(device)

    def extra_repr(self):
        return f'rerun={self.attention_rerun},use_pos={self.use_pos}'

    def embed(self, flat_src_tokens, **kwargs):
        assert not isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding)
        embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def embed_nodes(self, leave_embed, flat_node_tokens, **kwargs):
        if self.node_embed_init == 'embed':
            if self.args.pretrain_embed_mode == 'bert':
                return self.embed_tokens(flat_node_tokens, only_embedding=True)
            return self.embed(flat_node_tokens)
        elif self.node_embed_init == 'zero':
            b, l = flat_node_tokens.size()
            return torch.zeros(b, l, leave_embed.size(-1), dtype=leave_embed.dtype, device=leave_embed.device)
        else:
            raise ValueError(f'{self.node_embed_init} ???')

    @property
    def recusive_params(self):
        params = list(itertools.chain(*[x.recusive_params for x in self.layers]))
        return params

    def model_parallel_forward(self, src_node_leaves, src_node_nodes, src_node_indices, **kwargs):
        src_label_leaves = kwargs.get('src_label_leaves', None)
        b, n = src_node_leaves.size()
        b_, m = src_node_nodes.size()
        h = self.heads
        assert b == b_, f'{src_node_leaves.size()} != {src_node_nodes.size()}'
        leave_embeddings = self.embed(src_node_leaves)
        if self.use_pos:
            leave_embeddings += self.embed(src_label_leaves)
        node_embeddings = self.embed_nodes(leave_embeddings, src_node_nodes)

        leave_x = self.leave_embed_scale * leave_embeddings
        node_x = self.node_embed_scale * node_embeddings

        if self.pretrained_linear:
            leave_x = self.pretrained_proj(leave_x)

        if self.embed_positions is not None:
            leave_x += self.embed_positions(src_node_leaves)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.transpose(0, 1)
        node_x = node_x.transpose(0, 1)

        spans = src_node_indices

        key_pad = src_node_leaves.eq(self.padding_idx)
        node_pad = src_node_nodes.eq(self.padding_idx)
        if not key_pad.any() and not node_pad.any():
            key_pad = node_pad = None

        # build preliminaries
        device = leave_x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        pad_mask = self.nstack_mask_building_func(device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs)
        default_pad_mask = pad_mask if self.is_mask_default else self.nstack_default_mask_building_func(
            device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs
        )
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed else [None] * len(self.layers)

        if self.vanilla_layers > 0:
            leave = leave_x
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, key_pad)
            leave_x = leave

        attention_dict = {}
        assert self.attention_rerun == 1, f'self.attention_rerun = {self.attention_rerun}'
        for i, (layer, hier_embed, is_maskdf) in enumerate(zip(self.layers, hier_embeds, self.nstack_mask_df_layer)):
            pmask = default_pad_mask if is_maskdf else pad_mask
            try:
                if i > 0:
                    leave_x = leave_x.cuda(i)
                    node_x = node_x.cuda(i)
                    ntree_mask = ntree_mask.cuda(i)
                    hier_embed = hier_embed.cuda(i)
                    pmask = pmask.cuda(i)

                    key_pad = key_pad.cuda(i) if key_pad is not None else key_pad
                    node_pad = node_pad.cuda(i) if node_pad is not None else node_pad
                # print(f'iteration :{i}')
                leave_x, node_x, weights = layer(
                    leave_x, node_x, ntree_mask, hier_embed, pmask, key_pad, node_pad
                )
            except AssertionError as ae:
                print(f'Assert error at layer [{i}]')
                # src_node_leaves, src_node_nodes, src_node_indices
                print(f'sizes: {src_node_leaves.size()} / {src_node_nodes.size()} // {src_node_indices.size()}')
                torch.set_printoptions(profile="full")
                print(src_node_nodes)
                print(f'------------------------------')
                print(src_node_indices)
                print(f'------------------------------')
                torch.set_printoptions(profile="default")
                raise ae
            attention_dict[f'att_{i}'] = weights

        x = torch.cat((leave_x, node_x), 0)
        x = x.cuda(self.gpu_idx[-1])
        if self.normalize:
            x = self.layer_norm(x)

        out_dict = {
            'encoder_out': x,  # (n + m) x b x C
            'encoder_indices': src_node_indices,  # B x m x 2
            'encoder_padding_mask': key_pad,  # B x n
            'node_padding_mask': node_pad,  # B x m
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, **kwargs):
        """

        :param src_node_leaves:     [b, n]
        :param src_node_nodes:      [b, m]
        :param src_node_indices:    [b, m, 2]
        :param kwargs:
        :return:
        """
        if self.model_parallel:
            return self.model_parallel_forward(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        # assert (src_node_leaves < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_leaves.max()}'
        # assert (src_node_nodes < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_node_nodes.max()}'
        src_label_leaves = kwargs.get('src_label_leaves', None)
        b, n = src_node_leaves.size()
        b_, m = src_node_nodes.size()
        h = self.heads
        assert b == b_, f'{src_node_leaves.size()} != {src_node_nodes.size()}'
        leave_embeddings = self.embed(src_node_leaves)
        if self.use_pos:
            leave_embeddings += self.embed(src_label_leaves)
        node_embeddings = self.embed_nodes(leave_embeddings, src_node_nodes)

        leave_x = self.leave_embed_scale * leave_embeddings
        node_x = self.node_embed_scale * node_embeddings

        if self.pretrained_linear:
            leave_x = self.pretrained_proj(leave_x)

        if self.embed_positions is not None:
            leave_x += self.embed_positions(src_node_leaves)

        leave_x = self.embed_dropout_layer(leave_x)
        node_x = self.embed_dropout_layer(node_x)

        leave_x = leave_x.transpose(0, 1)
        node_x = node_x.transpose(0, 1)

        spans = src_node_indices

        key_pad = src_node_leaves.eq(self.padding_idx)
        node_pad = src_node_nodes.eq(self.padding_idx)
        if not key_pad.any() and not node_pad.any():
            key_pad = node_pad = None

        # build preliminaries
        device = leave_x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        pad_mask = self.nstack_mask_building_func(device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs)
        default_pad_mask = pad_mask if self.is_mask_default else self.nstack_default_mask_building_func(
            device, h, key_pad, node_pad, spans, b, n + m, n, m, **kwargs
        )
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed else [None] * len(self.layers)

        if self.vanilla_layers > 0:
            leave = leave_x
            for layer in self.vanilla_leave_layers:
                leave = layer(leave, key_pad)
            leave_x = leave

        attention_dict = {}
        for j in range(self.attention_rerun):
            for i, (layer, hier_embed, is_maskdf) in enumerate(zip(
                    self.layers, hier_embeds, self.nstack_mask_df_layer)):
                pmask = default_pad_mask if is_maskdf else pad_mask
                try:
                    leave_x, node_x, weights = layer(
                        leave_x, node_x, ntree_mask, hier_embed, pmask, key_pad, node_pad
                    )
                except AssertionError as ae:
                    print(f'Assert error at layer [{j}][{i}]')
                    # src_node_leaves, src_node_nodes, src_node_indices
                    print(f'sizes: {src_node_leaves.size()} / {src_node_nodes.size()} // {src_node_indices.size()}')
                    torch.set_printoptions(profile="full")
                    print(src_node_nodes)
                    print(f'------------------------------')
                    print(src_node_indices)
                    print(f'------------------------------')
                    torch.set_printoptions(profile="default")
                    raise ae
                attention_dict[f'att_{i}'] = weights

        x = torch.cat((leave_x, node_x), 0)
        if self.normalize:
            x = self.layer_norm(x)

        out_dict = {
            'encoder_out': x,  # (n + m) x b x C
            'encoder_indices': src_node_indices,  # B x m x 2
            'encoder_padding_mask': key_pad,  # B x n
            'node_padding_mask': node_pad,  # B x m
        }
        for k, v in attention_dict.items():
            out_dict[k] = v
        return out_dict

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['node_padding_mask'] is not None:
            encoder_out['node_padding_mask'] = encoder_out['node_padding_mask'].index_select(0, new_order)
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)
        for k in encoder_out.keys():
            if "att_" in k:
                encoder_out[k] = encoder_out[k].index_select(0, new_order)

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class NstackMerge2SeqTransformerDecoderLayer(nn.Module):

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.input_dropout = getattr(args, 'input_dropout', 0)

        self.relu_dropoute_layer = nn.Dropout(self.relu_dropout)
        self.plain_dropoute_layer = nn.Dropout(self.dropout)
        self.input_dropout_layer = nn.Dropout(self.input_dropout)

        self.normalize_before = args.decoder_normalize_before
        self.dptree_class = args.dptree_class
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)


        self.nstack_cross = getattr(args, 'nstack_cross', True)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
            cross_kwargs = {
                'nstack_mask_fn': getattr(args, 'cross_nstack_mask_fn', args.nstack_mask_fn)
            }
            if self.nstack_cross:
                print(f'Build Cross attention: {self.dptree_class}')
                self.encoder_attn = self.dptree_class(
                    args, self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout, **cross_kwargs
                )
            else:
                self.encoder_attn = MultiheadAttention(
                    self.embed_dim, args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def forward(
            self, x, enc_le, enc_no, ntree_mask, hier_embed, pad_mask, key_pad, node_pad,
            incremental_state,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None
            # x_le, x_no, ntree_mask, hier_embed, pad_mask, key_pad, node_pad
    ):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x = self.input_dropout_layer(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn = None
        need_weights = (not self.training and self.need_attn)
        # print(f'Cross:need_weights: {need_weights}/ {self.training} , {self.need_attn}')
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)
            if self.nstack_cross:
                # ntree_mask =
                # hier_embed =
                # pad_mask =
                x, attn = self.encoder_attn(
                    query=x,
                    key=enc_le, value=enc_le,
                    node_key=enc_no, node_value=enc_no,
                    ntree_mask=ntree_mask, hier_embed=hier_embed, pad_mask=pad_mask,
                    key_pad=key_pad, node_pad=node_pad, incremental_state=incremental_state,
                    need_weights=need_weights, static_kv=True
                )
            else:
                x, attn = self.encoder_attn(
                    query=x,
                    key=enc_le,
                    value=enc_le,
                    key_padding_mask=pad_mask,
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_weights,
                )
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.plain_dropoute_layer(x)
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.relu_dropoute_layer(x)
        x = self.fc2(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.plain_dropoute_layer(x)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn


class NstackMerge2SeqTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.embed_dropout_layer = nn.Dropout(self.dropout)

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        # TODO: set params
        assert args.encoder_embed_dim == args.decoder_embed_dim, f'encoder-decoder dim not work !='
        assert args.encoder_attention_heads == args.decoder_attention_heads, f'decoder_att_heads !='
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.nstack_cross = getattr(args, 'nstack_cross', True)
        self.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
        self.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
        self.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
        self.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)
        self.hier_pos_positions = MergeHierarchicalEmbedding(
            args, args.encoder_layers, self.head_dim, self.heads,
            self.nstack_hier_embed_max_horiz, self.nstack_hier_embed_max_ver,
            self.nstack_hier_embed_share
        ) if self.nstack_hier_embed and self.nstack_cross else None

        self.nstack_mask_fname = getattr(args, 'cross_nstack_mask_fn', args.nstack_mask_fn)
        self.mutual_level = getattr(args, 'mutual_ancestor_level', 5)
        self.nstack_mask_building_func = MergeWeightMask.acquire_mask_building_fn(
            self.nstack_mask_fname, self.mutual_level)

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            NstackMerge2SeqTransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.embed_dropout_layer(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # todo: retrieving cross attention
        # 'encoder_out': x,  # T x B x C
        # 'encoder_indices': src_node_indices,  # B x T x 2
        # 'encoder_padding_mask': key_padding_mask,  # B x T
        # 'node_padding_mask': node_padding_mask,  # B x T
        # out_dict = {
        #     'encoder_out': x,  # (n + m) x b x C
        #     'encoder_indices': src_node_indices,  # B x m x 2
        #     'encoder_padding_mask': key_pad,  # B x n
        #     'node_padding_mask': node_pad,  # B x m
        # }
        """
        :param key:                 [tk, b, m, c]
        :param value:               [tk, b, m, c]
        :param node_key:            [nk, b, m, c]
        :param node_value:          [nk, b, m, c]
        :param indices:             [nk, b, m, 2]
        """

        assert encoder_out is not None, f'encoder_out is None!'
        encoder_output = encoder_out['encoder_out']
        spans = encoder_out['encoder_indices']
        encoder_pad = encoder_out['encoder_padding_mask']
        node_pad = encoder_out['node_padding_mask']

        nm, b_, c = encoder_output.size()
        b, m, _ = spans.size()
        # tk = tnk - nk
        h = self.heads
        n = nm - m
        encoder_leaves = encoder_output[:n]
        encoder_nodes = encoder_output[n:]

        inner_atts = []

        # ntree_mask =
        # hier_embed =
        # pad_mask =

        device = x.device
        ntree_mask = MergeStackNodesOnAffinityValueAttention.get_ntree_mask(n, spans, self.heads)
        pad_mask = self.nstack_mask_building_func(device, h, encoder_pad, node_pad, spans, b, x.size(0), n, m, **kwargs)
        hier_embeds = self.hier_pos_positions(n, spans) if self.nstack_hier_embed and self.nstack_cross else [None] * len(self.layers)
        for i, (layer, hier_embed) in enumerate(zip(self.layers, hier_embeds)):
            # hier_embed = hier_embeds[i]
            x, attn = layer(
                x=x, enc_le=encoder_leaves, enc_no=encoder_nodes,
                ntree_mask=ntree_mask, hier_embed=hier_embed, pad_mask=pad_mask, key_pad=encoder_pad, node_pad=node_pad,
                incremental_state=incremental_state,
                prev_self_attn_state=None,
                prev_attn_state=None,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)
            inner_atts.append(attn)

        if self.normalize:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states, 'inner_atts': inner_atts}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict






