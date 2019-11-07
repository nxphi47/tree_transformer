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

from fairseq.models import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel, FairseqModel, register_model, FairseqEncoderModel,
    register_model_architecture,
)

from ..data.dptree_sep_mono_class_dataset import *
from .. import utils as src_utils

# from .tree_transformer_layers import *
from ..modules.dptree_multihead_attention import *
from ..modules.dptree_sep_multihead_attention import *
from ..modules.dptree_individual_multihead_attention import *
from ..modules.dptree_onseq_multihead_attention import *

from ..modules.dptree_transformer_layer import *
from ..modules.embeddings import *

from ..modules.nstack_transformer_layers import *
from ..modules.nstack_tree_attention import *
from ..modules.nstack_merge_tree_attention import *

from fairseq.models.transformer import *


class _NstackProjector(nn.Module):

    def __init__(self, projector):
        super().__init__()
        self.projector = projector

    def forward(self, encoder_out):
        pass


class FirstRootNstackProjector(_NstackProjector):

    def forward(self, encoder_out):
        # (t + n) x B x m x C
        class_token = encoder_out[-1, :, 0]
        class_output = self.projector(class_token)
        return class_output


class SeqRootFirstNstackProjector(_NstackProjector):

    def forward(self, encoder_out):
        class_token = encoder_out
        class_output = self.class_projector(class_token)
        class_output = class_output.permute(1, 2, 0, 3)
        class_output = torch.flip(class_output, [2])
        return class_output


@register_model('nstack_encoder')
class NstackTransformerEncoderClass(FairseqEncoderModel):
    def __init__(self, args, encoder, class_projector):
        super().__init__(encoder)
        self.args = args
        self.return_first_root = getattr(args, 'return_first_root', False)
        self.dropout_class = getattr(args, 'dropout_class', False)
        self.class_projector = class_projector
        self.dropout_class_layer = nn.Dropout(args.dropout if self.dropout_class else 0.0)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--pretrain_embed_mode', default='const', help='pretrain_embed_mode')
        parser.add_argument('--tune_epoch', type=int, metavar='N', default=100000000)
        parser.add_argument('--attention_rerun', type=int, metavar='N', default=1)
        # parser.add_argument('--return_type', type=str, default='first_root')
        parser.add_argument('--return_first_root', default=False, action='store_true',)
        parser.add_argument('--dropout_class', default=False, action='store_true',)
        parser.add_argument('--nclasses', type=int, metavar='N', default=2, help='number of classes')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    @property
    def recusive_params(self):
        try:
            params = self.encoder.recusive_params
        except Exception as e:
            params = []
        return params

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs):
        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs)
        # (t + n) x B x m x C
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out[-1, :, 0]

        class_output = self.class_projector(class_token)
        return class_output


@register_model('nstack_nli_encoder')
class NstackTransformerNLIEncoderClass(NstackTransformerEncoderClass):

    @staticmethod
    def add_args(parser):
        NstackTransformerEncoderClass.add_args(parser)
        parser.add_argument('--concat', default=False, action='store_true', )

    @property
    def is_concat(self):
        return self.args.concat

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_in_dim = args.encoder_embed_dim if args.concat else args.encoder_embed_dim * 2
        class_projector = Linear(class_in_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def _forward_noconcat(
            self,
            src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1,
            src_node_leaves2, src_node_nodes2, src_node_indices2, src_sent_lengths2,
            **kwargs):
        enc_output1 = self.encoder(src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1, **kwargs)
        enc_output2 = self.encoder(src_node_leaves2, src_node_nodes2, src_node_indices2, src_sent_lengths2, **kwargs)
        # (t + n) x B x m x C
        encoder_out1 = enc_output1['encoder_out']
        encoder_out2 = enc_output2['encoder_out']
        class_token1 = encoder_out1[-1, :, 0]
        class_token2 = encoder_out2[-1, :, 0]
        class_token = torch.cat([class_token1, class_token2], -1)
        class_output = self.class_projector(class_token)
        return class_output

    def _forward_concat(
            self,
            src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1,
            src_node_leaves2, src_node_nodes2, src_node_indices2, src_sent_lengths2,
            **kwargs):
        assert src_node_leaves2 is None
        assert src_node_nodes2 is None
        assert src_node_indices2 is None
        assert src_sent_lengths2 is None
        enc_output1 = self.encoder(src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1, **kwargs)
        encoder_out1 = enc_output1['encoder_out']
        class_token1 = encoder_out1[-1, :, 0]
        class_output = self.class_projector(class_token1)
        return class_output

    def forward(
            self,
            src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1,
            src_node_leaves2=None, src_node_nodes2=None, src_node_indices2=None, src_sent_lengths2=None,
            **kwargs):

        try:
            if self.is_concat:
                return self._forward_concat(
                    src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1,
                    src_node_leaves2, src_node_nodes2, src_node_indices2, src_sent_lengths2,
                )
            else:
                return self._forward_noconcat(
                    src_node_leaves1, src_node_nodes1, src_node_indices1, src_sent_lengths1,
                    src_node_leaves2, src_node_nodes2, src_node_indices2, src_sent_lengths2,
                )
        except RuntimeError as eruntime:
            if 'out of memory' in str(eruntime):
                ls = src_node_leaves1.size()
                ns = src_node_nodes1.size()
                ls2 = src_node_leaves2.size() if src_node_leaves2 is not None else None
                ns2 = src_node_nodes2.size() if src_node_nodes2 is not None else None
                print(f'| WARNING-FORWARD: [{ls},{ns}]/[{ls2},{ns2}] OOM exception: {str(eruntime)};\n Skipping batch')
            raise eruntime


@register_model('nstack_node_encoder')
class NstackNodeTransformerEncoderClass(NstackTransformerEncoderClass):
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_in_dim = args.encoder_embed_dim if args.concat else args.encoder_embed_dim * 2
        class_projector = Linear(class_in_dim, args.nclasses)
        # class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs):
        # print(f'Batch {src_tokens.size()}')
        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs)
        # (t + n) x B x m x C
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out

        if self.return_first_root:
            class_token = encoder_out[-1, :, 0]
            class_token = self.dropout_class_layer(class_token)
            class_output = self.class_projector(class_token)
            return class_output
        else:
            class_token = self.dropout_class_layer(class_token)
            class_output = self.class_projector(class_token)
            class_output = class_output.permute(1, 2, 0, 3)
            class_output = torch.flip(class_output, [2])
            # now the top should be the first root
            return class_output


@register_model('nstack_flatseq_node_encoder')
class NstackNodeFlatSeqTransformerEncoderClass(NstackNodeTransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_in_dim = args.encoder_embed_dim if args.concat else args.encoder_embed_dim * 2
        class_projector = Linear(class_in_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs):
        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs)
        # (t + n) x B x m x C
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out

        if self.return_first_root:
            class_token = encoder_out[-1, :, 0]
            class_token = self.dropout_class_layer(class_token)
            class_output = self.class_projector(class_token)
            return class_output
        else:
            # class_token = self.dropout_class_layer(class_token)
            # class_output = self.class_projector(class_token)
            # class_output = class_output.permute(1, 2, 0, 3)
            # class_output = torch.flip(class_output, [2])
            # now the top should be the first root
            # return class_output
            raise ValueError(f'must be return_first_root')


@register_model('nstack2seq')
class Nstack2SeqTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        assert not args.left_pad_source, f'args.left_pad_source = {args.left_pad_source}, should not be True'
        args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)
        args.decoder_type = getattr(args, 'decoder_type', Nstack2SeqTransformerDecoder)
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = args.decoder_type(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, prev_output_tokens, **kwargs):
        try:
            encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, src_sent_lengths, **kwargs)
            assert encoder_output is not None, f'encoder_out is None!'
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_output, **kwargs)
        except RuntimeError as er:
            if 'out of memory' in str(er):
                ls = src_node_leaves.size()
                ns = src_node_nodes.size()
                print(f'| WARNING-FORWARD: [{ls},{ns}] OOM exception: {str(er)};\n Skipping batch')
            raise er
        return decoder_out


@register_model('nstack_merge2seq')
class NstackMerge2SeqTransformer(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--use_pos', default=False, action='store_true')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        assert not args.left_pad_source, f'args.left_pad_source = {args.left_pad_source}, should not be True'
        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
        args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = args.decoder_type(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, prev_output_tokens, **kwargs):
        try:
            encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
            assert encoder_output is not None, f'encoder_out is None!'
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_output, **kwargs)
        except RuntimeError as er:
            if 'out of memory' in str(er):
                ls = src_node_leaves.size()
                ns = src_node_nodes.size()
                print(f'| WARNING-FORWARD: [{ls},{ns}] OOM exception: {str(er)};\n Skipping batch')
            raise er
        return decoder_out


@register_model('nstack_merge2seq_fake')
class NstackMerge2SeqFakeTransformer(NstackMerge2SeqTransformer):
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        assert not args.left_pad_source, f'args.left_pad_source = {args.left_pad_source}, should not be True'
        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
        args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
        # encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        # decoder = args.decoder_type(args, tgt_dict, decoder_embed_tokens)
        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(encoder, decoder)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, prev_output_tokens, **kwargs):
        try:
            src_lens = (~src_node_leaves.eq(self.encoder.padding_idx)).int().sum(-1)
            encoder_output = self.encoder(src_node_leaves, src_lens)
            assert encoder_output is not None, f'encoder_out is None!'
            decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_output)
        except RuntimeError as er:
            if 'out of memory' in str(er):
                ls = src_node_leaves.size()
                ns = src_node_nodes.size()
                print(f'| WARNING-FORWARD: [{ls},{ns}] OOM exception: {str(er)};\n Skipping batch')
            raise er
        return decoder_out


import torch
import torch.nn as nn
import torch.nn.functional as F
class MP(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(10, 10).cuda(0)
        self.b = nn.Linear(10, 10).cuda(1)
    def forward(self, x, y):
        x = x.cuda(0)
        x = self.a(x)
        x = self.b(x.cuda(1))
        # loss = (x * y.to(x.device)).sum()
        x = F.log_softmax(x, -1)
        loss = F.nll_loss(x, y.to(x.device), size_average=False, reduce=True)
        return loss

# c = MP()
# x = torch.Tensor(100, 10).uniform_().cuda(0)
# y = (torch.Tensor(100).uniform_() * 10).long().cuda(0)
# loss = c(x, y)

@register_model('nstack_merge_encoder')
class NstackMergeTransformerEncoderClass(NstackTransformerEncoderClass):

    def __init__(self, args, encoder, class_projector):
        super().__init__(args, encoder, class_projector)
        self.gpu_idx = None
        self.model_parallel = False

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--pretrain_embed_mode', default='const', help='pretrain_embed_mode')
        parser.add_argument('--tune_epoch', type=int, metavar='N', default=100000000)
        parser.add_argument('--attention_rerun', type=int, metavar='N', default=1)
        # parser.add_argument('--return_type', type=str, default='first_root')
        parser.add_argument('--return_first_root', default=False, action='store_true', )
        parser.add_argument('--dropout_class', default=False, action='store_true', )
        parser.add_argument('--nclasses', type=int, metavar='N', default=5, help='number of classes')

    def setup_cuda(self, gpu_idx):
        self.gpu_idx = gpu_idx
        self.model_parallel = True
        # self.class_projector = class_projector
        # self.dropout_class_layer = nn.Dropout(args.dropout if self.dropout_class else 0.0)
        # self.encoder = encoder
        self.encoder.setup_cuda(self.gpu_idx)
        self.class_projector.cuda(self.gpu_idx[-1])
        print(f'|| [{self.gpu_idx[-1]}][class_projector]: {self.class_projector}')
        self.dropout_class_layer.cuda(self.gpu_idx[-1])
        print(f'|| [{self.gpu_idx[-1]}][dropout_class_layer]: {self.dropout_class_layer}')
        return self

    def cuda(self, device=None):
        return super().cuda(device)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        # class_in_dim = args.encoder_embed_dim if args.concat else args.encoder_embed_dim * 2
        # class_projector = Linear(class_in_dim, args.nclasses)
        class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=True to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, **kwargs):
        # print(f'{src_node_leaves.size()} - {src_node_nodes.size()} - {src_node_indices.size()}')
        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        # (t + n) x B x m x C
        encoder_out = encoder_output['encoder_out']

        # class_token = encoder_out[-1]
        # class_token = self.dropout_class_layer(class_token)
        # class_output = self.class_projector(class_token)

        if self.return_first_root:
            class_token = encoder_out[-1]
            class_token = self.dropout_class_layer(class_token)
            class_output = self.class_projector(class_token)
            return class_output
        else:
            # fix_here
            class_token = self.dropout_class_layer(encoder_out)
            class_output = self.class_projector(class_token)
            class_output = class_output.transpose(0, 1)
            # class_output = torch.flip(class_output, [2])
            # now the top should be the first root
            return class_output


@register_model('nstack_merge_lm_encoder')
class NstackMergeLMTransformerEncoderClass(NstackMergeTransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
        assert not args.left_pad_source, f'Must LEFT_PAD=True to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_node_leaves, src_node_nodes, src_node_indices, **kwargs):
        # print(f'{src_node_leaves.size()} - {src_node_nodes.size()} - {src_node_indices.size()}')
        n = src_node_leaves.size(1)
        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        # (t + n) x B x m x C
        encoder_out = encoder_output['encoder_out']
        node_out = encoder_out[n:]
        node_out = F.linear(node_out, self.encoder.embed_tokens.weight)

        class_token = self.dropout_class_layer(encoder_out)
        class_output = self.class_projector(class_token)
        class_output = class_output.transpose(0, 1)
        # class_output = torch.flip(class_output, [2])
        # now the top should be the first root
        return class_output, node_out



# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = lvec * rvec
        abs_dist = torch.abs(lvec - rvec)
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = torch.sigmoid(self.wh(vec_dist))
        # out = F.log_softmax(self.wp(out), dim=1)
        out = self.wp(out)
        return out


@register_model('nstack_merge_relateness_encoder')
class NstackMergeRelatenessTransformerEncoderClass(NstackTransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        # class_projector = Linear(args.encoder_embed_dim, args.nclasses)
        class_projector = Similarity(args.encoder_embed_dim, args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=True to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(
            self, src_node_leaves, src_node_nodes, src_node_indices,
            src2_node_leaves, src2_node_nodes, src2_node_indices,
            **kwargs):
        # print(f'{src_node_leaves.size()} - {src_node_nodes.size()} - {src_node_indices.size()}')
        # encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        # (t + n) x B x m x C
        # encoder_out = encoder_output['encoder_out']

        encoder_output_a = self.encoder(src_node_leaves, src_node_nodes, src_node_indices, **kwargs)
        encoder_output_b = self.encoder(src2_node_leaves, src2_node_nodes, src2_node_indices, **kwargs)
        # (t + n) x B x m x C
        encout_a = encoder_output_a['encoder_out']
        encout_b = encoder_output_b['encoder_out']

        tok_a = encout_a[-1]
        tok_b = encout_b[-1]

        out = self.class_projector(tok_a, tok_b)
        return out


@register_model('nstack_merge_relateness_concat_encoder')
class NstackMergeRelatenessTransformerEncoderClass(NstackTransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        if args.share_all_embeddings:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_transformer_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim, args.nclasses)
        # class_projector = Similarity(args.encoder_embed_dim, args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=True to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(
            self, src_node_leaves, src_node_nodes, src_node_indices, src_mask_le, src_mask_no,
            **kwargs):

        encoder_output = self.encoder(src_node_leaves, src_node_nodes, src_node_indices,
                                        src_mask_le=src_mask_le, src_mask_no=src_mask_no, **kwargs)
        # (t + n) x B x m x C
        encout = encoder_output['encoder_out']
        out = self.class_projector(encout[-1])
        return out


@register_model_architecture('nstack_encoder', 'nstack_class_base')
def nstack_class_base(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackOnValueAttention)
    args.placeholder_const = getattr(args, 'placeholder_const', False)
    args.pretrain_embed_mode = getattr(args, 'pretrain_embed_mode', 'const')
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)

    args.src_len_norm = getattr(args, 'src_len_norm', 'none')
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)

    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'none')

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.DEFAULT)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', None)

    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', False)

    args.take_full_dim = getattr(args, 'take_full_dim', False)
    args.hier_embed_right = getattr(args, 'hier_embed_right', False)

    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'none')
    args.node_embed_init = getattr(args, 'node_embed_init', 'embed')

    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', False)

    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', False)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 0)

    args.transition_act = getattr(args, 'transition_act', 'none')
    args.transition_dropout = getattr(args, 'transition_dropout', 0.0)

    args.mutual_ancestor_level = getattr(args, 'mutual_ancestor_level', 5)
    args.sep_dwstack_proj_act = getattr(args, 'sep_dwstack_proj_act', 'tanh')

    args.nstack_cross = getattr(args, 'nstack_cross', True)

    args.input_dropout = getattr(args, 'input_dropout', 0)
    base_architecture(args)


def add_s2s_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 32)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 32)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 32)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 32)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 2)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)


def add_iwslt(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)


def add_vaswani_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)


def add_hier_full_dim(args):
    args.take_full_dim = getattr(args, 'take_full_dim', False)


@register_model_architecture('nstack2seq', 'nstack2seq_base')
def nstack2seq_base(args):
    nstack_class_base(args)


@register_model_architecture('nstack_merge2seq_fake', 'nstack2seq_merge_fake_base')
def nstack2seq_merge_fake_base(args):
    nstack_class_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_onkey_base')
def nstack_class_onkey_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackOnKeyAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_base')
def nstack_node_class_onkey_base(args):
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_base')
def nstack_node_class_base(args):
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_base')
def wnstack_node_class_onaff_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackWeightedOnAffinityAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_base')
def wnstack_node_class_onvalue_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackWeightedOnValueAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_base')
def dwnstack_node_class_onaff_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_base')
def dwnstack_node_class_onvalue_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onkey_base')
def dwnstack_node_class_onkey_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnKeyAttention)
    nstack_class_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaff_base')
def dwnstack2seq_node_onaff_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityAttention)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onvalue_base')
def dwnstack2seq_node_onvalue_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onkey_base')
def dwnstack2seq_node_onkey_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnKeyAttention)
    nstack2seq_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_base')
def nstack_node_class_sumup_onkey_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackSumUpOnKeyAttention)
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_base')
def nstack_node_class_sumup_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackSumUpOnValueAttention)
    nstack_class_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_base')
def nstack_node_class_sumup_onvalue_base(args):
    nstack_node_class_sumup_base(args)


def add_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)


def add_ffn_512(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 512)


def add_ffn_1024(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 1024)


def add_ffn_256(args):
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 256)


def add_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    add_tiny(args)


def add_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    add_tiny(args)


def add_tiny_v5(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    add_tiny(args)


def add_tiny_v6(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    add_tiny(args)


def add_tiny_v7(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 2)
    add_tiny(args)


def add_tiny_v3_ffn512(args):
    add_ffn_512(args)
    add_tiny_v3(args)


def add_tiny_v3_ffn1024(args):
    add_ffn_1024(args)
    add_tiny_v3(args)


def _flatten_target_programs(iterable):
    yield "["
    it = iter(iterable)
    for e in it:
        if isinstance(e, (list, tuple)):
            for f in _flatten_target_programs(e):
                yield f
        else:
            yield e
    yield "]"
import json
def _parse_json_to_dict(json_line):
    line_json_dict = json.loads(json_line)
    # The features of interest "text" and "short_tree" are stored as lists in
    # this dictionary -- "short_tree" is a nested list. We flatten and join the
    # lists on space, to return a string in both these cases.
    # Make another dictionary, to return only the features we want.
    return {
        "inputs":
            " ".join(line_json_dict["text"]),
        "targets":
            " ".join([
                i for i in _flatten_target_programs(
                    line_json_dict["short_tree"])
            ])
    }

def write(file, prefix):
    with open(file, 'r') as f:
        lines = f.read().strip().split('\n')
    inputs = []
    targets = []
    for l in lines:
        obj = _parse_json_to_dict(l)
        inputs.append(obj['inputs'])
        targets.append(obj['targets'])
    with open(f'{prefix}.in', 'w') as f:
        f.write('\n'.join(inputs))
    with open(f'{prefix}.out', 'w') as f:
        f.write('\n'.join(targets))


def add_tiny_v4(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    add_tiny(args)


def add_tiny_v4_ffn512(args):
    add_ffn_512(args)
    add_tiny_v4(args)


def add_tiny_v7h4ffn256(args):
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    add_ffn_256(args)
    add_tiny_v7(args)


def add_tiny_d128f256(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 128)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 256)
    add_tiny(args)


# def add_tiny_v5_ffn512(args):
#     add_ffn_512(args)
#     add_tiny_v4(args)


def add_tiny_300d(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 10)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    add_tiny(args)


def add_tiny_300d_v2(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    add_tiny(args)


@register_model_architecture('nstack_encoder', 'nstack_class_onkey_tiny')
def nstack_class_onkey_tiny(args):
    add_tiny(args)
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_onkey_tiny_v2')
def nstack_class_onkey_tiny_v2(args):
    add_tiny_v2(args)
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_onkey_tiny_v3')
def nstack_class_onkey_tiny_v3(args):
    add_tiny_v3(args)
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_onkey_tiny_300d')
def nstack_class_onkey_tiny_300d(args):
    add_tiny_300d(args)
    nstack_class_onkey_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_tiny')
def nstack_class_tiny(args):
    add_tiny(args)
    nstack_class_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_tiny_v2')
def nstack_class_tiny_v2(args):
    add_tiny_v2(args)
    nstack_class_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_tiny_v3')
def nstack_class_tiny_v3(args):
    add_tiny_v3(args)
    nstack_class_base(args)


@register_model_architecture('nstack_encoder', 'nstack_class_tiny_300d')
def nstack_class_tiny_300d(args):
    add_tiny_300d(args)
    nstack_class_base(args)

# 0000 NODE


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny')
def nstack_node_class_onkey_tiny(args):
    add_tiny(args)
    nstack_node_class_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_v2')
def nstack_node_class_onkey_tiny_v2(args):
    add_tiny_v2(args)
    nstack_node_class_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_v3')
def nstack_node_class_onkey_tiny_v3(args):
    add_tiny_v3(args)
    nstack_node_class_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_300d')
def nstack_node_class_onkey_tiny_300d(args):
    add_tiny_300d(args)
    nstack_node_class_onkey_base(args)

# asdasd
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny')
def wnstack_node_class_onaff_tiny(args):
    add_tiny(args)
    wnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_v2')
def wnstack_node_class_onaff_tiny_v2(args):
    add_tiny_v2(args)
    wnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_v3')
def wnstack_node_class_onaff_tiny_v3(args):
    add_tiny_v3(args)
    wnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_v3_ffn512')
def wnstack_node_class_onaff_tiny_v3_ffn512(args):
    add_tiny_v3_ffn512(args)
    wnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_v3_ffn512')
def dwnstack_node_class_onaff_tiny_v3_ffn512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_v4_ffn512')
def dwnstack_node_class_onaff_tiny_v4_ffn512(args):
    add_tiny_v4_ffn512(args)
    dwnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_v5')
def dwnstack_node_class_onaff_tiny_v5(args):
    add_tiny_v5(args)
    dwnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny')
def dwnstack_node_class_onaff_tiny(args):
    add_tiny(args)
    dwnstack_node_class_onaff_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_300d')
def wnstack_node_class_onaff_tiny_300d(args):
    add_tiny_300d(args)
    wnstack_node_class_onaff_base(args)

# asdasd
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny')
def wnstack_node_class_onvalue_tiny(args):
    add_tiny(args)
    wnstack_node_class_onvalue_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_v2')
def wnstack_node_class_onvalue_tiny_v2(args):
    add_tiny_v2(args)
    wnstack_node_class_onvalue_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_v3')
def wnstack_node_class_onvalue_tiny_v3(args):
    add_tiny_v3(args)
    wnstack_node_class_onvalue_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_v3_ffn512')
def wnstack_node_class_onvalue_tiny_v3_ffn512(args):
    add_tiny_v3_ffn512(args)
    wnstack_node_class_onvalue_base(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_300d')
def wnstack_node_class_onvalue_tiny_300d(args):
    add_tiny_300d(args)
    wnstack_node_class_onvalue_base(args)


# -- pos_embedding

@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_posemb')
def nstack_node_class_onkey_tiny_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_cummean')
def nstack_node_class_onkey_tiny_cummean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_posemb_cummean')
def nstack_node_class_onkey_tiny_posemb_cummean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_posemb_cumsqrtmean')
def nstack_node_class_onkey_tiny_posemb_cumsqrtmean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_linear')
def nstack_node_class_onkey_tiny_linear(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', True)
    nstack_node_class_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_linear_posemb')
def nstack_node_class_onkey_tiny_linear_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    nstack_node_class_onkey_tiny_linear(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_linear_posemb_cummean')
def nstack_node_class_onkey_tiny_linear_posemb_cummean(args):
    args.cum_node = getattr(args, 'cum_node', 'mean')
    nstack_node_class_onkey_tiny_linear_posemb(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tiny_linear_posemb_cumsqrtmean')
def nstack_node_class_onkey_tiny_linear_posemb_cumsqrtmean(args):
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    nstack_node_class_onkey_tiny_linear_posemb(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_onkey_tanhup_tiny_posemb_cummean')
def nstack_node_class_onkey_tanhup_tiny_posemb_cummean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackOnKeyTanhUpAttention)
    nstack_node_class_onkey_tiny_posemb_cummean(args)



# @register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_linear_posemb_cumsqrtmean')
# def nstack_node_class_sumup_onkey_tiny_linear_posemb_cumsqrtmean(args):
#     args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
#     # NodeStackSumUpOnValueAttention
#     args.dptree_class = getattr(args, 'dptree_class', NodeStackSumUpOnKeyAttention)
#     nstack_node_class_onkey_tiny_linear_posemb(args)

# todo: Sum-up family
@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny')
def nstack_node_class_sumup_onkey_tiny(args):
    add_tiny(args)
    nstack_node_class_sumup_onkey_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_posemb')
def nstack_node_class_sumup_onkey_tiny_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_cummean')
def nstack_node_class_sumup_onkey_tiny_cummean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_posemb_cumsqrtmean')
def nstack_node_class_sumup_onkey_tiny_posemb_cumsqrtmean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_linear')
def nstack_node_class_sumup_onkey_tiny_linear(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', True)
    nstack_node_class_sumup_onkey_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_linear_posemb')
def nstack_node_class_sumup_onkey_tiny_linear_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    nstack_node_class_sumup_onkey_tiny_linear(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onkey_tiny_linear_posemb_cummean')
def nstack_node_class_sumup_onkey_tiny_linear_posemb_cummean(args):
    args.cum_node = getattr(args, 'cum_node', 'mean')
    nstack_node_class_sumup_onkey_tiny_linear_posemb(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny')
def nstack_node_class_sumup_onvalue_tiny(args):
    add_tiny(args)
    nstack_node_class_sumup_onvalue_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_posemb')
def nstack_node_class_sumup_onvalue_tiny_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onvalue_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_cummean')
def nstack_node_class_sumup_onvalue_tiny_cummean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onvalue_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean')
def nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.nstack_linear = getattr(args, 'nstack_linear', False)
    nstack_node_class_sumup_onvalue_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_linear')
def nstack_node_class_sumup_onvalue_tiny_linear(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', False)
    args.nstack_pos_embed_learned = getattr(args, 'nstack_pos_embed_learned', False)
    args.cum_node = getattr(args, 'cum_node', 'sum')
    args.nstack_linear = getattr(args, 'nstack_linear', True)
    nstack_node_class_sumup_onvalue_tiny(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_linear_posemb')
def nstack_node_class_sumup_onvalue_tiny_linear_posemb(args):
    args.nstack_pos_embed = getattr(args, 'nstack_pos_embed', True)
    nstack_node_class_sumup_onvalue_tiny_linear(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_linear_posemb_cummean')
def nstack_node_class_sumup_onvalue_tiny_linear_posemb_cummean(args):
    args.cum_node = getattr(args, 'cum_node', 'mean')
    nstack_node_class_sumup_onvalue_tiny_linear_posemb(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean_v2')
def nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean_v2(args):
    add_tiny_v2(args)
    nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean_v3')
def nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean_v3(args):
    add_tiny_v3(args)
    nstack_node_class_sumup_onvalue_tiny_posemb_cumsqrtmean(args)




# weighted-on-aff
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_noincleaves_v3')
def wnstack_node_class_onaff_tiny_noincleaves_v3(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    wnstack_node_class_onaff_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_sqrtmean_v3')
def wnstack_node_class_onaff_tiny_sqrtmean_v3(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onaff_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_mean_v3')
def wnstack_node_class_onaff_tiny_mean_v3(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    wnstack_node_class_onaff_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3')
def wnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onaff_tiny_v3(args)


# weighted-on-aff
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_noincleaves_v3')
def wnstack_node_class_onvalue_tiny_noincleaves_v3(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    wnstack_node_class_onvalue_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_sqrtmean_v3')
def wnstack_node_class_onvalue_tiny_sqrtmean_v3(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onvalue_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_mean_v3')
def wnstack_node_class_onvalue_tiny_mean_v3(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    wnstack_node_class_onvalue_tiny_v3(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3')
def wnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onvalue_tiny_v3(args)



# weighted-on-aff
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_noincleaves_v3_ffn512')
def wnstack_node_class_onaff_tiny_noincleaves_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    wnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_sqrtmean_v3_ffn512')
def wnstack_node_class_onaff_tiny_sqrtmean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_mean_v3_ffn512')
def wnstack_node_class_onaff_tiny_mean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    wnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3_ffn512')
def wnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onaff_tiny_v3_ffn512(args)


# weighted-on-aff
@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_noincleaves_v3_ffn512')
def wnstack_node_class_onvalue_tiny_noincleaves_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    wnstack_node_class_onvalue_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_sqrtmean_v3_ffn512')
def wnstack_node_class_onvalue_tiny_sqrtmean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onvalue_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_mean_v3_ffn512')
def wnstack_node_class_onvalue_tiny_mean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    wnstack_node_class_onvalue_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'wnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3_ffn512')
def wnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    wnstack_node_class_onvalue_tiny_v3_ffn512(args)

#
# # weighted-on-aff
# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_noincleaves_v3_ffn512')
# def dwnstack_node_class_onvalue_tiny_noincleaves_v3_ffn512(args):
#     args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
#     dwnstack_node_class_onvalue_tiny_v3_ffn512(args)
#
#
# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_v3_ffn512')
# def dwnstack_node_class_onvalue_tiny_sqrtmean_v3_ffn512(args):
#     # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
#     dwnstack_node_class_onvalue_tiny_v3_ffn512(args)
#
#
# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_mean_v3_ffn512')
# def dwnstack_node_class_onvalue_tiny_mean_v3_ffn512(args):
#     # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
#     dwnstack_node_class_onvalue_tiny_v3_ffn512(args)
#
#
# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3_ffn512')
# def dwnstack_node_class_onvalue_tiny_noincleaves_sqrtmean_v3_ffn512(args):
#     args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
#     dwnstack_node_class_onvalue_tiny_v3_ffn512(args)
#





# weighted-on-aff
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_noincleaves_v3_ffn512')
def dwnstack_node_class_onaff_tiny_noincleaves_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_sqrtmean_v3_ffn512')
def dwnstack_node_class_onaff_tiny_sqrtmean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_mean_v3_ffn512')
def dwnstack_node_class_onaff_tiny_mean_v3_ffn512(args):
    # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3_ffn512')
def dwnstack_node_class_onaff_tiny_noincleaves_sqrtmean_v3_ffn512(args):
    args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)











@register_model_architecture('nstack_node_encoder', 'nstack_node_class_tiny')
def nstack_node_class_tiny(args):
    add_tiny(args)
    nstack_node_class_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_tiny_v2')
def nstack_node_class_tiny_v2(args):
    add_tiny_v2(args)
    nstack_node_class_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_tiny_v3')
def nstack_node_class_tiny_v3(args):
    add_tiny_v3(args)
    nstack_node_class_base(args)


@register_model_architecture('nstack_node_encoder', 'nstack_node_class_tiny_300d')
def nstack_node_class_tiny_300d(args):
    add_tiny_300d(args)
    nstack_node_class_base(args)

















