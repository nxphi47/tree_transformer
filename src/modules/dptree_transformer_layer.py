import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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

DEBUG = False


def print_debug(s):
    if DEBUG:
        print('dptree_transformer_layer.py::' + s)


class DPTreeTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class

        self.self_attn = self.dptree_class(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def forward(self, x, indices, flat_indices, gt_indices, encoder_padding_mask):
        """

        :param x:   input tensor        `(seq_len, batch, embed_dim)` ?????
        :param indices: indices         `(seq_len, batch, 2)`
        :param flat_indices:            [B * h, d, Tk]
        :param gt_indices:              [B * h, Tq, Tk]
        :param encoder_padding_mask:
        :return: encoded output shape
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)

        # TODO--here
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            flat_indices=flat_indices,
            gt_indices=gt_indices,
            key_padding_mask=encoder_padding_mask
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x


class DPTree2SeqTransformerDecoderLayer(nn.Module):
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

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # self.dptree_class = args.dptree_class
            # self.encoder_attn = DPTreeMultiheadAttention(
            self.encoder_attn = self.dptree_class(
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
            encoder_out,
            encoder_padding_mask,
            encoder_indices,
            encoder_flat_indices,
            encoder_gt_indices,
            incremental_state,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None):
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
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                # indices=encoder_indices,
                flat_indices=encoder_flat_indices,
                gt_indices=encoder_gt_indices,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
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


class DPTreeTransformerEncoder(FairseqEncoder):

    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

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
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        # ) if not args.token_positional_embeddings else None
        # ) if args.encoder_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DPTreeTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_indices (LongTensor): `(batch, src_len, 2)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions

        # compute padding mask
        # print(f'size:src_tokens={src_tokens.size()}, src_indices={src_indices.size()}, src_lengths={src_lengths.cpu()}')
        # print(f'device:src_tokens={src_tokens.device}, src_indices={src_indices.device}, src_lengths={src_lengths.device}')

        try:
            # print_debug(f'device: - {src_tokens.device} - {src_indices.device} - {src_lengths.device} || {self.embed_tokens.num_embeddings}')
            # print_debug(f'src_indices===')
            # print_debug(f'{src_indices}')
            # print_debug(f'src_tokens===')
            # print_debug(f'{src_tokens}')

            # if (src_tokens >= self.embed_tokens.num_embeddings).any():
            #     print(f'src_tokens has id >= vocab')
            #     print(f'{src_tokens}')
            #     raise ValueError(f'src_tokens >= num_embeddings!')
            # elif (src_tokens > 0).all():
            #     print(f'src_tokens < 1')
            #     print(f'{src_tokens}')
            #     raise ValueError(f'src_tokens > 0 failed!')
            # else:
            assert (
                    src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= num_embeddings:max= {src_tokens.max()}'

            x = self.embed_scale * self.embed_tokens(src_tokens)
            if self.embed_positions is not None:
                x += self.embed_positions(src_tokens)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

            encoder_padding_mask = src_tokens.eq(self.padding_idx)
            if not encoder_padding_mask.any():
                encoder_padding_mask = None
        except Exception as e:
            # maybe_print(f'Something Fail!, lengths {src_lengths}')
            # # maybe_print(f'size:src_tokens={src_tokens.size()}, src_indices={src_indices.size()}, src_lengths={src_lengths.cpu()}')
            # maybe_print(f'device:src_tokens={src_tokens.device}, src_indices={src_indices.device}, src_lengths={src_lengths.device}')
            print(f'{self.__class__.__name__} forward failed!')
            print(f'-------src_indices--------')
            # _idx = src_indices.cpu().numpy()
            # _src = src_tokens.cpu().numpy()
            # _len = src_lengths.cpu().numpy()
            print(f'{src_indices}')
            print(f'-------src_lengths--------')
            print(f'{src_lengths}')
            print(f'-------src_tokens--------')
            print(f'{src_tokens}')

            # print(f'-------encoder_padding_mask--------')
            # print(f'{encoder_padding_mask}')
            raise e

        src_size = src_tokens.size()
        node_len = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        # src_flat_indices = DPTreeMultiheadAttention.indices2flat_indices(
        #     src_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads)
        #
        # # self-attention node_len
        # src_gt_indices = DPTreeMultiheadAttention.indices2gt_indices(
        #     src_indices, seq_len=seq_len, query_len=node_len, heads=self.heads,
        # )
        src_flat_indices = self.dptree_class.indices2flat_indices(
            src_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=node_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, seq_len=seq_len, query_len=node_len, heads=self.heads, head_dim=self.head_dim
        )

        # encoder layers
        for layer in self.layers:
            # x, indices, flat_indices, gt_indices, encoder_padding_mask
            x = layer(x, src_indices, src_flat_indices, src_gt_indices, encoder_padding_mask)

        if self.normalize:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_indices': src_indices,  # B x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

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
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)

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


class DPTree2SeqTransformerDecoder(FairseqIncrementalDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

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
            DPTree2SeqTransformerDecoderLayer(args, no_encoder_attn)
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
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # TODO: build flat and gt indices
        assert encoder_out is not None, f'encoder_out is None!'
        encoder_indices = encoder_out['encoder_indices']
        encoder_size = encoder_indices.size()
        node_len = encoder_size[1]
        seq_len = int(((node_len + 1) // 2) + 1)
        tgt_len = x.size(0)
        # encoder_flat_indices = DPTreeMultiheadAttention.indices2flat_indices(
        #     encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads
        # )
        # encoder_gt_indices = DPTreeMultiheadAttention.indices2gt_indices(
        #     encoder_indices, seq_len=seq_len, query_len=x.size()[0], heads=self.heads
        # )
        #
        encoder_flat_indices = self.dptree_class.indices2flat_indices(
            encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=tgt_len
        )
        encoder_gt_indices = self.dptree_class.indices2gt_indices(
            encoder_indices, seq_len=seq_len, query_len=tgt_len, heads=self.heads, head_dim=self.head_dim
        )

        # decoder layers
        for layer in self.layers:
            """
            x, 
            encoder_out, 
            encoder_padding_mask, 
            encoder_indices,
            encoder_flat_indices,
            encoder_gt_indices,
            """
            x, attn = layer(
                x=x,
                encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                encoder_indices=encoder_out['encoder_indices'] if encoder_out is not None else None,
                encoder_flat_indices=encoder_flat_indices,
                encoder_gt_indices=encoder_gt_indices,
                incremental_state=incremental_state,
                # x, encoder_out, encoder_padding_mask, encoder_indices, incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

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

        return x, {'attn': attn, 'inner_states': inner_states}

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


class DPTreeSeparateTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class

        # self.self_attn = DPTreeMultiheadAttention(
        self.self_attn = self.dptree_class(
            args, self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def forward(self, x, flat_indices, gt_indices, encoder_padding_mask):
        """

        :param x:                       [tq, b, m, C]
        :param flat_indices:            [B * h, m, Tq, Tk]
        :param gt_indices:              [B * h, m, Tq, Tk]
        :param encoder_padding_mask:
        :return: encoded output shape
        """
        residual = x
        assert not torch.isnan(x).any()

        x = self.maybe_layer_norm(0, x, before=True)

        tq, bsz, nsent, dim = x.size()
        query = x.transpose(1, 2).contiguous().view(tq * nsent, bsz, dim)
        assert not torch.isnan(x).any(), f'x problem'
        assert not torch.isnan(query).any(), f'query problem'

        # TODO--here
        x, _ = self.self_attn(
            query=query, key=x, value=x,
            flat_indices=flat_indices,
            gt_indices=gt_indices,
            key_padding_mask=encoder_padding_mask,
            force_self_att=True
        )
        assert not torch.isnan(x).any(), f'after self attention problem'

        # tq, nsent, new_dim = x.size()
        x = x.contiguous().view(tq, nsent, bsz, dim).transpose(1, 2)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        return x


class DPTreeSeparateTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

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
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DPTreeSeparateTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def embed(self, flat_src_tokens, flat_indices):
        assert not torch.isnan(flat_src_tokens).any()
        if isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding):
            embeddings = self.embed_tokens(flat_src_tokens, flat_indices)
        else:
            embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """

        :param src_tokens:      [b, m, src_len]
        :param src_indices:     [b, m, src_len, 2]
        :param src_lengths:     [b, m]
        :param kwargs:
        :return:
        """
        assert (src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_tokens.max()}'

        bsz, nsent, src_len = src_tokens.size()

        flat_src_tokens = src_tokens.view(bsz * nsent, src_len)
        flat_indices = src_indices.view(bsz * nsent, src_len, 2)
        assert not torch.isnan(flat_src_tokens).any()
        embeddings = self.embed(flat_src_tokens, flat_indices)

        assert not torch.isnan(embeddings).any(), f'{self.embed_tokens.num_embeddings} ??? {flat_src_tokens.max()} ??? {flat_src_tokens.min()}'
        x = self.embed_scale * embeddings

        assert not torch.isnan(x).any(), f'[{self.embed_tokens.num_embeddings}][{torch.isnan(x).int().sum()}]::: {flat_src_tokens}'

        if self.embed_positions is not None:
            x += self.embed_positions(flat_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(bsz, nsent, src_len, x.size(-1)).permute(2, 0, 1, 3)
        # [tq, b, m, C]

        # print(f'x:dim = {x.size()}')

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        else:
            assert not encoder_padding_mask[:, 0, 0].any(), f'{encoder_padding_mask}'

        # [b, m, tk]
        src_size = src_tokens.size()
        node_len = src_size[2]
        nsent = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        query_len = node_len * nsent

        src_fl_indices = self.dptree_class.indices2flat_indices(
            src_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=query_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, nsent=nsent, seq_len=seq_len, query_len=query_len, heads=self.heads, head_dim=self.head_dim
        )

        for i, layer in enumerate(self.layers):
            # x, indices, flat_indices, gt_indices, encoder_padding_mask
            # print(f'Layer {i}.....')

            x = layer(x, src_fl_indices, src_gt_indices, encoder_padding_mask)
            assert not torch.isnan(x).any()

        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim

        return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }

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
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)

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


class DPTree2SeqSeparateTransformerDecoderLayer(nn.Module):
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

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            # self.dptree_class = args.dptree_class
            # self.encoder_attn = DPTreeMultiheadAttention(
            self.encoder_attn = self.dptree_class(
                args, self.embed_dim, args.decoder_attention_heads,
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
            encoder_out,
            encoder_padding_mask,
            encoder_indices,
            encoder_flat_indices,
            encoder_gt_indices,
            incremental_state,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None):
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
        if self.encoder_attn is not None:
            residual = x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            if prev_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_attn_state
                saved_state = {"prev_key": prev_key, "prev_value": prev_value}
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                # indices=encoder_indices,
                flat_indices=encoder_flat_indices,
                gt_indices=encoder_gt_indices,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
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


class DPTree2SeqSeparateTransformerDecoder(FairseqIncrementalDecoder):
    """
    return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }
    """
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

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
            DPTree2SeqSeparateTransformerDecoderLayer(args, no_encoder_attn)
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
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # TODO: build flat and gt indices
        assert encoder_out is not None, f'encoder_out is None!'
        encoder_indices = encoder_out['encoder_indices']
        encoder_size = encoder_indices.size()

        nsent = encoder_size[1]
        node_len = encoder_size[2]
        seq_len = int(((node_len + 1) // 2) + 1)
        tgt_len = x.size(0)
        # encoder_flat_indices = DPTreeMultiheadAttention.indices2flat_indices(
        #     encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads
        # )
        # encoder_gt_indices = DPTreeMultiheadAttention.indices2gt_indices(
        #     encoder_indices, seq_len=seq_len, query_len=x.size()[0], heads=self.heads
        # )
        #
        encoder_flat_indices = self.dptree_class.indices2flat_indices(
            # encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=tgt_len,
            encoder_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=tgt_len

        )
        encoder_gt_indices = self.dptree_class.indices2gt_indices(
            # encoder_indices, seq_len=seq_len, query_len=tgt_len, heads=self.heads, head_dim=self.head_dim
            encoder_indices, nsent=nsent, seq_len=seq_len, query_len=tgt_len, heads=self.heads, head_dim=self.head_dim
        )

        # decoder layers
        for layer in self.layers:
            """
            x, 
            encoder_out, 
            encoder_padding_mask, 
            encoder_indices,
            encoder_flat_indices,
            encoder_gt_indices,
            """
            x, attn = layer(
                x=x,
                encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                encoder_indices=encoder_out['encoder_indices'] if encoder_out is not None else None,
                encoder_flat_indices=encoder_flat_indices,
                encoder_gt_indices=encoder_gt_indices,
                incremental_state=incremental_state,
                # x, encoder_out, encoder_padding_mask, encoder_indices, incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

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

        return x, {'attn': attn, 'inner_states': inner_states}

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


class DPTree2SeqSeparateNoDPTCrossTransformerDecoder(FairseqIncrementalDecoder):
    """
        return {
                'encoder_out': x,  # T x B x m x C
                'encoder_indices': src_indices,  # B x m x T x 2
                'encoder_padding_mask': encoder_padding_mask,  # B x m x T
            }
        """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

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
            TransformerDecoderLayer(args, no_encoder_attn)
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
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # TODO: build flat and gt indices
        assert encoder_out is not None, f'encoder_out is None!'
        encoder_indices = encoder_out['encoder_indices']
        encoder_size = encoder_indices.size()

        # nsent = encoder_size[1]
        # node_len = encoder_size[2]
        # seq_len = int(((node_len + 1) // 2) + 1)
        # tgt_len = x.size(0)
        # encoder_flat_indices = DPTreeMultiheadAttention.indices2flat_indices(
        #     encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads
        # )
        # encoder_gt_indices = DPTreeMultiheadAttention.indices2gt_indices(
        #     encoder_indices, seq_len=seq_len, query_len=x.size()[0], heads=self.heads
        # )
        #
        # encoder_flat_indices = self.dptree_class.indices2flat_indices(
        #     # encoder_indices, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=tgt_len,
        #     encoder_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=tgt_len
        #
        # )
        # encoder_gt_indices = self.dptree_class.indices2gt_indices(
        #     # encoder_indices, seq_len=seq_len, query_len=tgt_len, heads=self.heads, head_dim=self.head_dim
        #     encoder_indices, nsent=nsent, seq_len=seq_len, query_len=tgt_len, heads=self.heads, head_dim=self.head_dim
        # )

        # decoder layers
        # for layer in self.layers:
        #     """
        #     x,
        #     encoder_out,
        #     encoder_padding_mask,
        #     encoder_indices,
        #     encoder_flat_indices,
        #     encoder_gt_indices,
        #     """
        #     x, attn = layer(
        #         x=x,
        #         encoder_out=encoder_out['encoder_out'] if encoder_out is not None else None,
        #         encoder_padding_mask=encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
        #         encoder_indices=encoder_out['encoder_indices'] if encoder_out is not None else None,
        #         encoder_flat_indices=encoder_flat_indices,
        #         encoder_gt_indices=encoder_gt_indices,
        #         incremental_state=incremental_state,
        #         # x, encoder_out, encoder_padding_mask, encoder_indices, incremental_state,
        #         self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
        #     )
        #     inner_states.append(x)

        if encoder_out is not None:
            # 'encoder_out': x,  # T x B x m x C
            # 'encoder_indices': src_indices,  # B x m x T x 2
            # 'encoder_padding_mask': encoder_padding_mask,  # B x m x T
            encoder_output = encoder_out['encoder_out']
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            tk, b, m, c = encoder_output.size()

            if encoder_padding_mask is not None:
                _b, _m, _tk = encoder_padding_mask.size()
                assert b == _b, f'{b} != {_b}'
                assert m == _m, f'{m} != {_m}'
                assert tk == _tk, f'{tk} != {_tk}'
                encoder_padding_mask = encoder_padding_mask.view(b, m * tk)

            encoder_output = encoder_output.permute(2, 0, 1, 3).contiguous().view(m * tk, b, c)
        else:
            encoder_output = None
            encoder_padding_mask = None

        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_output,
                encoder_padding_mask,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

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

        return x, {'attn': attn, 'inner_states': inner_states}

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


class DPTreeSeparateRootAverageTransformerEncoder(DPTreeSeparateTransformerEncoder):

    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """

        :param src_tokens:      [b, m, src_len]
        :param src_indices:     [b, m, src_len, 2]
        :param src_lengths:     [b, m]
        :param kwargs:
        :return:
        """
        assert (src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_tokens.max()}'

        bsz, nsent, src_len = src_tokens.size()

        flat_src_tokens = src_tokens.view(bsz * nsent, src_len)
        flat_indices = src_indices.view(bsz * nsent, src_len, 2)
        assert not torch.isnan(flat_src_tokens).any()
        embeddings = self.embed(flat_src_tokens, flat_indices)

        assert not torch.isnan(
            embeddings).any(), f'{self.embed_tokens.num_embeddings}?{flat_src_tokens.max()}?{flat_src_tokens.min()}'
        x = self.embed_scale * embeddings

        assert not torch.isnan(
            x).any(), f'[{self.embed_tokens.num_embeddings}][{torch.isnan(x).int().sum()}]::: {flat_src_tokens}'

        if self.embed_positions is not None:
            x += self.embed_positions(flat_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(bsz, nsent, src_len, x.size(-1)).permute(2, 0, 1, 3)
        # [tq, b, m, C]

        # print(f'x:dim = {x.size()}')

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        else:
            assert not encoder_padding_mask[:, 0, 0].any(), f'{encoder_padding_mask}'

        # [b, m, tk]
        src_size = src_tokens.size()
        node_len = src_size[2]
        nsent = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        query_len = node_len * nsent

        src_fl_indices = self.dptree_class.indices2flat_indices(
            src_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=query_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, nsent=nsent, seq_len=seq_len, query_len=query_len, heads=self.heads, head_dim=self.head_dim
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, src_fl_indices, src_gt_indices, encoder_padding_mask)
            assert not torch.isnan(x).any()

        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim
        # FIXME: reduce all to only the root of the tree and average them.
        # FIXME: improper use will lead to error
        x = x[:1]
        src_indices = src_indices[:, :, :1]
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask[:, :, :1]
            sent_mask = (~encoder_padding_mask).permute(2, 0, 1).unsqueeze(-1).type_as(x)
            sent_mask_sum = sent_mask.sum(dim=2, keepdim=True)
        else:
            sent_mask = 1.0
            sent_mask_sum = float(x.size(2))
        # x:    [1, b, m, c]
        # i:    [b, m, 1, c]
        # mask: [1, b, m, 1]

        x = (x * sent_mask).sum(dim=2, keepdim=True) / sent_mask_sum

        return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }


class DPTreeIndividualTransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.dptree_class = args.dptree_class

        # self.self_attn = DPTreeMultiheadAttention(
        self.self_attn = self.dptree_class(
            args, self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

    def forward(self, x, flat_indices, gt_indices, encoder_padding_mask):
        """

        :param x:                       [tq, b, m, C]
        :param flat_indices:            [B * h, m, Tq, Tk]
        :param gt_indices:              [B * h, m, Tq, Tk]
        :param encoder_padding_mask:
        :return: encoded output shape
        """
        residual = x
        assert not torch.isnan(x).any()

        x = self.maybe_layer_norm(0, x, before=True)

        tq, bsz, nsent, dim = x.size()
        # query = x.transpose(1, 2).contiguous().view(tq * nsent, bsz, dim)
        assert not torch.isnan(x).any(), f'x problem'
        # assert not torch.isnan(query).any(), f'query problem'

        # TODO--here
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            flat_indices=flat_indices,
            gt_indices=gt_indices,
            key_padding_mask=encoder_padding_mask,
            force_self_att=True
        )
        assert not torch.isnan(x).any(), f'after self attention problem'

        tq, nsent, bsz, dim = x.size()
        x = x.transpose(1, 2)
        # x = x.contiguous().transpose(1, 2)
        # x:        [tq, b, m, d]

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem'
        x = self.maybe_layer_norm(1, x, after=True)
        assert not torch.isnan(x).any(), f'after maybe_layer_norm problem 2'

        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class DPTreeIndividualTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

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
        # self.head_dim = self.encoder_embed_dim // self.head_dim

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DPTreeIndividualTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])
        self.tfm_layers = nn.ModuleList([])
        self.tfm_layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def embed(self, flat_src_tokens, flat_indices):
        assert not torch.isnan(flat_src_tokens).any()
        if isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding):
            embeddings = self.embed_tokens(flat_src_tokens, flat_indices)
        else:
            embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings

    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """

        :param src_tokens:      [b, m, src_len]
        :param src_indices:     [b, m, src_len, 2]
        :param src_lengths:     [b, m]
        :param kwargs:
        :return:
        """
        assert (src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_tokens.max()}'

        bsz, nsent, src_len = src_tokens.size()

        flat_src_tokens = src_tokens.view(bsz * nsent, src_len)
        flat_indices = src_indices.view(bsz * nsent, src_len, 2)
        assert not torch.isnan(flat_src_tokens).any()
        embeddings = self.embed(flat_src_tokens, flat_indices)

        assert not torch.isnan(embeddings).any(), f'{self.embed_tokens.num_embeddings} ??? {flat_src_tokens.max()} ??? {flat_src_tokens.min()}'
        x = self.embed_scale * embeddings

        assert not torch.isnan(x).any(), f'[{self.embed_tokens.num_embeddings}][{torch.isnan(x).int().sum()}]::: {flat_src_tokens}'

        if self.embed_positions is not None:
            x += self.embed_positions(flat_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(bsz, nsent, src_len, x.size(-1)).permute(2, 0, 1, 3)
        # [tq, b, m, C]

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # encoder_padding_mask:     [b, m, src_len]
        # linear_enc_pad_mask:      [b, src_len * m]
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
            linear_enc_pad_mask = None
        else:
            assert not encoder_padding_mask[:, 0, 0].any(), f'{encoder_padding_mask}'
            linear_enc_pad_mask = encoder_padding_mask.transpose(1, 2).contiguous().view(bsz, src_len * nsent)

        # [b, m, tk]
        src_size = src_tokens.size()
        node_len = src_size[2]
        nsent = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        # query_len = node_len * nsent
        query_len = node_len

        src_fl_indices = self.dptree_class.indices2flat_indices(
            src_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=query_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, nsent=nsent, seq_len=seq_len, query_len=query_len, heads=self.heads, head_dim=self.head_dim
        )

        for i, (layer, linear_layer) in enumerate(zip(self.layers, self.tfm_layers)):
            x = layer(x, src_fl_indices, src_gt_indices, encoder_padding_mask)
            # tq, bsz, nsent, new_dim
            # [tq, b, m, d]
            tq, bsz, m, d = x.size()
            assert not torch.isnan(x).any()
            """x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``."""

            linear_x = x.transpose(1, 2).contiguous().view(tq * m, bsz, d)
            # linear_x:             [tq * m, b, d]
            # linear_enc_pad_mask   [b, tq * m]
            if linear_enc_pad_mask is not None:
                assert linear_enc_pad_mask.size(1) == linear_x.size(0), f'{linear_enc_pad_mask.size()} != {linear_x.size()}'

            linear_x = linear_layer(linear_x, linear_enc_pad_mask)
            # linear_x:             [tq * m, b, d]
            x = linear_x.contiguous().view(tq, m, bsz, d).transpose(1, 2)

        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim

        return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }

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
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)

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


class DPTreeIndividualRootAverageTransformerEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens, left_pad=False):
        super().__init__(dictionary)
        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        # TODO: what is this?
        self.max_source_positions = args.max_source_positions
        # TODO: set params
        assert not left_pad
        self.heads = args.encoder_attention_heads
        self.encoder_embed_dim = args.encoder_embed_dim
        self.head_dim = self.encoder_embed_dim // self.heads

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            left_pad=left_pad,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.dptree_class = args.dptree_class

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DPTreeIndividualTransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.encoder_normalize_before
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def embed(self, flat_src_tokens, flat_indices):
        assert not torch.isnan(flat_src_tokens).any()
        if isinstance(self.embed_tokens, PhraseAveragePretrainedEmbedding):
            embeddings = self.embed_tokens(flat_src_tokens, flat_indices)
        else:
            embeddings = self.embed_tokens(flat_src_tokens)
        return embeddings


    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """

        :param src_tokens:      [b, m, src_len]
        :param src_indices:     [b, m, src_len, 2]
        :param src_lengths:     [b, m]
        :param kwargs:
        :return:
        """
        assert (src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_tokens.max()}'

        bsz, nsent, src_len = src_tokens.size()

        flat_src_tokens = src_tokens.view(bsz * nsent, src_len)
        flat_indices = src_indices.view(bsz * nsent, src_len, 2)
        assert not torch.isnan(flat_src_tokens).any()
        embeddings = self.embed(flat_src_tokens, flat_indices)

        assert not torch.isnan(embeddings).any(), f'{self.embed_tokens.num_embeddings} ??? {flat_src_tokens.max()} ??? {flat_src_tokens.min()}'
        x = self.embed_scale * embeddings

        assert not torch.isnan(x).any(), f'[{self.embed_tokens.num_embeddings}][{torch.isnan(x).int().sum()}]::: {flat_src_tokens}'

        if self.embed_positions is not None:
            x += self.embed_positions(flat_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(bsz, nsent, src_len, x.size(-1)).permute(2, 0, 1, 3)
        # [tq, b, m, C]

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # encoder_padding_mask:     [b, m, src_len]
        # linear_enc_pad_mask:      [b, src_len * m]
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
            # linear_enc_pad_mask = None
        else:
            assert not encoder_padding_mask[:, 0, 0].any(), f'{encoder_padding_mask}'
            # linear_enc_pad_mask = encoder_padding_mask.transpose(1, 2).contiguous().view(bsz, src_len * nsent)

        # [b, m, tk]
        src_size = src_tokens.size()
        node_len = src_size[2]
        nsent = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        # query_len = node_len * nsent
        query_len = node_len

        src_fl_indices = self.dptree_class.indices2flat_indices(
            src_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=query_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, nsent=nsent, seq_len=seq_len, query_len=query_len, heads=self.heads, head_dim=self.head_dim
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, src_fl_indices, src_gt_indices, encoder_padding_mask)
            # tq, bsz, nsent, new_dim
            # [tq, b, m, d]
            tq, bsz, m, d = x.size()
            assert not torch.isnan(x).any()
            """x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``."""

            # linear_x = x.transpose(1, 2).contiguous().view(tq * m, bsz, d)
            # # linear_x:             [tq * m, b, d]
            # # linear_enc_pad_mask   [b, tq * m]
            # if linear_enc_pad_mask is not None:
            #     assert linear_enc_pad_mask.size(1) == linear_x.size(0), f'{linear_enc_pad_mask.size()} != {linear_x.size()}'
            #
            # linear_x = linear_layer(linear_x, linear_enc_pad_mask)
            # # linear_x:             [tq * m, b, d]
            # x = linear_x.contiguous().view(tq, m, bsz, d).transpose(1, 2)

        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim
        # FIXME: reduce all to only the root of the tree and average them.
        # FIXME: improper use will lead to error
        # x = x[:1]
        # src_indices = src_indices[:, :, :1]
        # encoder_padding_mask = encoder_padding_mask[:, :, :1]
        # sent_mask = (~encoder_padding_mask).permute(2, 0, 1).unsqueeze(-1).type_as(x)
        x = x[:1]
        src_indices = src_indices[:, :, :1]
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask[:, :, :1]
            sent_mask = (~encoder_padding_mask).permute(2, 0, 1).unsqueeze(-1).type_as(x)
            sent_mask_sum = sent_mask.sum(dim=2, keepdim=True)
        else:
            sent_mask = 1.0
            sent_mask_sum = float(x.size(2))
        # x:    [1, b, m, c]
        # i:    [b, m, 1, c]
        # mask: [1, b, m, 1]

        x = (x * sent_mask).sum(dim=2, keepdim=True) / sent_mask_sum

        return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }

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
        if encoder_out['encoder_indices'] is not None:
            encoder_out['encoder_indices'] = encoder_out['encoder_indices'].index_select(0, new_order)

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


class DPTreeIndividualNoAverageTransformerEncoder(DPTreeIndividualRootAverageTransformerEncoder):

    def forward(self, src_tokens, src_indices, src_lengths, **kwargs):
        """

        :param src_tokens:      [b, m, src_len]
        :param src_indices:     [b, m, src_len, 2]
        :param src_lengths:     [b, m]
        :param kwargs:
        :return:
        """
        assert (src_tokens < self.embed_tokens.num_embeddings).all(), f'src_tokens >= :max= {src_tokens.max()}'

        bsz, nsent, src_len = src_tokens.size()

        flat_src_tokens = src_tokens.view(bsz * nsent, src_len)
        flat_indices = src_indices.view(bsz * nsent, src_len, 2)
        assert not torch.isnan(flat_src_tokens).any()
        embeddings = self.embed(flat_src_tokens, flat_indices)

        assert not torch.isnan(embeddings).any(), f'{self.embed_tokens.num_embeddings} ??? {flat_src_tokens.max()} ??? {flat_src_tokens.min()}'
        x = self.embed_scale * embeddings

        assert not torch.isnan(x).any(), f'[{self.embed_tokens.num_embeddings}][{torch.isnan(x).int().sum()}]::: {flat_src_tokens}'

        if self.embed_positions is not None:
            x += self.embed_positions(flat_src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(bsz, nsent, src_len, x.size(-1)).permute(2, 0, 1, 3)
        # [tq, b, m, C]

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # encoder_padding_mask:     [b, m, src_len]
        # linear_enc_pad_mask:      [b, src_len * m]
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
            # linear_enc_pad_mask = None
        else:
            assert not encoder_padding_mask[:, 0, 0].any(), f'{encoder_padding_mask}'
            # linear_enc_pad_mask = encoder_padding_mask.transpose(1, 2).contiguous().view(bsz, src_len * nsent)

        # [b, m, tk]
        src_size = src_tokens.size()
        node_len = src_size[2]
        nsent = src_size[1]
        seq_len = int((node_len + 1) // 2) + 1

        # query_len = node_len * nsent
        query_len = node_len

        src_fl_indices = self.dptree_class.indices2flat_indices(
            src_indices, nsent=nsent, seq_len=seq_len, head_dim=self.head_dim, heads=self.heads, query_len=query_len
        )

        # self-attention node_len
        src_gt_indices = self.dptree_class.indices2gt_indices(
            src_indices, nsent=nsent, seq_len=seq_len, query_len=query_len, heads=self.heads, head_dim=self.head_dim
        )

        for i, layer in enumerate(self.layers):
            x = layer(x, src_fl_indices, src_gt_indices, encoder_padding_mask)
            # tq, bsz, nsent, new_dim
            # [tq, b, m, d]
            tq, bsz, m, d = x.size()
            assert not torch.isnan(x).any()

        if self.normalize:
            x = self.layer_norm(x)

        # tq, bsz, nsent, new_dim

        return {
            'encoder_out': x,  # T x B x m x C
            'encoder_indices': src_indices,  # B x m x T x 2
            'encoder_padding_mask': encoder_padding_mask,  # B x m x T
        }










