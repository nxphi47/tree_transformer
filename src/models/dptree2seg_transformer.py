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
from fairseq.models.transformer import *


@register_model('dptree2seq')
class DPTree2SeqTransformer(TransformerModel):

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
            # print(f'Dict: {dictionary} PadID: {padding_idx} - size = {len(dictionary)}')
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
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
        # encoder = LearnableGlobalLocalTransformerEncoder(
        #     args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        encoder = DPTreeTransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = DPTree2SeqTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return DPTree2SeqTransformer(encoder, decoder)

    def forward(self, src_tokens, src_labels, src_indices, src_sent_lengths, src_lengths, prev_output_tokens, **kwargs):
        try:
            encoder_out = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)
            decoder_out = self.decoder(prev_output_tokens, encoder_out, **kwargs)
        except RuntimeError as er:
            if 'out of memory' in str(er):
                print(f'| WARNING-FORWARD: [{src_tokens.size()}] ran out of memory with exception: {str(er)};\n Skipping batch')
            raise er
        return decoder_out


@register_model('dptree2seq_sep')
class DPTree2SeqSeparateTransformer(DPTree2SeqTransformer):
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
            # print(f'Dict: {dictionary} PadID: {padding_idx} - size = {len(dictionary)}')
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
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
        # encoder = LearnableGlobalLocalTransformerEncoder(
        #     args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
        args.decoder_type = getattr(args, 'decoder_type', DPTree2SeqSeparateTransformerDecoder)
        encoder = DPTreeSeparateTransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        decoder = args.decoder_type(args, tgt_dict, decoder_embed_tokens)
        return DPTree2SeqSeparateTransformer(encoder, decoder)


@register_model('dptree_encoder')
class DPTreeTransformerEncoderClass(FairseqEncoderModel):
    def __init__(self, encoder, class_projector):
        super().__init__(encoder)
        self.class_projector = class_projector

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--nclasses', type=int, metavar='N', default=2,
                            help='number of classes')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            # if src_dict != tgt_dict:
            #     raise ValueError('--share-all-embeddings requires a joined dictionary')
            # if args.encoder_embed_dim != args.decoder_embed_dim:
            #     raise ValueError(
            #         '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            # if args.decoder_embed_path and (
            #         args.decoder_embed_path != args.encoder_embed_path):
            #     raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            # decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        args.encoder_type = getattr(args, 'encoder_type', DPTreeTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=True to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return DPTreeTransformerEncoderClass(encoder, class_projector)

    def forward(self, src_tokens, src_labels, src_indices, src_sent_lengths, src_lengths, **kwargs):

        encoder_output = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out[0]
        class_output = self.class_projector(class_token)
        # class_token:  [b, C]
        # C = 128
        # self.max_positions()
        return class_output


@register_model('dptree_sep_encoder')
class DPTreeSeparateTransformerEncoderClass(FairseqEncoderModel):
    def __init__(self, args, encoder, class_projector):
        super().__init__(encoder)
        self.args = args
        self.return_first_root = getattr(args, 'return_first_root', False)
        self.class_projector = class_projector

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--pretrain_embed_mode', default='const', help='pretrain_embed_mode')
        parser.add_argument('--tune_epoch', type=int, metavar='N', default=60)
        # parser.add_argument('--nclasses', type=int, metavar='N', default=2, help='number of classes')
        parser.add_argument('--return_first_root', default=False, action='store_true', )
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

        args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_tokens, src_labels, src_indices, src_sent_lengths, src_lengths, **kwargs):
        # print(f'Batch {src_tokens.size()}')
        encoder_output = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out[0, :, 0]
        class_output = self.class_projector(class_token)
        return class_output


@register_model('dptree_sep_node_encoder')
class DPTreeSeparateNodeTransformerEncoderClass(DPTreeSeparateTransformerEncoderClass):

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
        args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(self, src_tokens, src_labels, src_indices, src_sent_lengths, src_lengths, **kwargs):
        encoder_output = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)

        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out

        if self.return_first_root:
            class_token = encoder_out[0, :, 0]
            class_output = self.class_projector(class_token)
            return class_output
        else:
            class_output = self.class_projector(class_token)
            class_output = class_output.permute(1, 2, 0, 3)
            return class_output


@register_model('dptree_sep_nli_encoder')
class DPTreeSeparateNLITransformerEncoderClass(DPTreeSeparateTransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)

            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim * 2, args.nclasses)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = DPTreeSeparateTransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def forward(
            self,
            src_tokens1, src_labels1, src_indices1, src_sent_lengths1, src_lengths1,
            src_tokens2, src_labels2, src_indices2, src_sent_lengths2, src_lengths2,
            **kwargs):

        encoder_output1 = self.encoder(src_tokens1, src_indices1, src_lengths1, **kwargs)
        encoder_output2 = self.encoder(src_tokens2, src_indices2, src_lengths2, **kwargs)

        encoder_out1 = encoder_output1['encoder_out']
        encoder_out2 = encoder_output2['encoder_out']

        class_token1 = encoder_out1[0, :, 0]
        class_token2 = encoder_out2[0, :, 0]

        overall_class_token = torch.cat([class_token1, class_token2], -1)
        class_output = self.class_projector(overall_class_token)
        return class_output


@register_model('dptree_sep_nli_concat_encoder')
class DPTreeSeparateNLIConcatTransformerEncoderClass(DPTreeSeparateNLITransformerEncoderClass):

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict = task.source_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)

            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        class_projector = Linear(args.encoder_embed_dim, args.nclasses)

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = DPTreeSeparateTransformerEncoder(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return cls(args, encoder, class_projector)

    def merge(self, x1, x2, dim=2):
        ndims = x1.dim()
        assert ndims == x2.dim()
        length = max(x1.size(2), x2.size(2))
        offset_1 = length - x1.size(2)
        offset_2 = length - x2.size(2)
        x1 = F.pad(x1, [0, 0] * (ndims - 3) + [0, offset_1, 0, 0, 0, 0])
        x2 = F.pad(x2, [0, 0] * (ndims - 3) + [0, offset_2, 0, 0, 0, 0])
        return x1, x2

    def forward(
            self,
            src_tokens1, src_labels1, src_indices1, src_sent_lengths1, src_lengths1,
            src_tokens2, src_labels2, src_indices2, src_sent_lengths2, src_lengths2,
            **kwargs):
        # print(f'before-{src_tokens1.size()},{src_tokens2.size()}, {src_indices1.size()}, {src_indices2.size()}')
        src_tokens1, src_tokens2 = self.merge(src_tokens1, src_tokens2)
        src_indices1, src_indices2 = self.merge(src_indices1, src_indices2)
        # print(f'after-{src_tokens1.size()},{src_tokens2.size()}, {src_indices1.size()}, {src_indices2.size()}')

        src_tokens = torch.cat([src_tokens1, src_tokens2], 1)
        # src_labels = torch.cat([src_labels1, src_labels2], 1)
        src_indices = torch.cat([src_indices1, src_indices2], 1)
        src_lengths = src_lengths1 + src_lengths2

        encoder_output = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)

        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out[0, :, 0]
        class_output = self.class_projector(class_token)
        return class_output


@register_model('dptree_indi_encoder')
class DPTreeIndividualTransformerEncoderClass(FairseqEncoderModel):
    def __init__(self, encoder, class_projector):
        super().__init__(encoder)
        self.class_projector = class_projector

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--encoder-token-positional-embeddings', default=False, action='store_true',
            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--pretrain_embed_mode', default='const', help='pretrain_embed_mode')
        parser.add_argument('--tune_epoch', type=int, metavar='N', default=60)
        parser.add_argument('--nclasses', type=int, metavar='N', default=2, help='number of classes')

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualTransformerEncoder)

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

        assert not args.left_pad_source, f'Must LEFT_PAD=False to get the first(root) state'
        encoder = args.encoder_type(args, src_dict, encoder_embed_tokens, left_pad=args.left_pad_source)
        return DPTreeIndividualTransformerEncoderClass(encoder, class_projector)

    def forward(self, src_tokens, src_labels, src_indices, src_sent_lengths, src_lengths, **kwargs):
        print(f'Batch {src_tokens.size()}')
        encoder_output = self.encoder(src_tokens, src_indices, src_lengths, **kwargs)
        encoder_out = encoder_output['encoder_out']
        class_token = encoder_out[0, :, 0]
        class_output = self.class_projector(class_token)
        return class_output


@register_model_architecture('dptree2seq', 'dptree2seq')
def dptree2seq_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeMultiheadAttention)
    args.placeholder_const = getattr(args, 'placeholder_const', False)
    args.pretrain_embed_mode = getattr(args, 'pretrain_embed_mode', 'const')
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)
    args.src_len_norm = getattr(args, 'src_len_norm', 'sqrt')
    base_architecture(args)


@register_model_architecture('dptree2seq', 'dptree2seq_onlyk')
def dptree2seq_onlyk_base(args):
    args.dptree_class = getattr(args, 'dptree_class', DPTreeOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_base')
def dptree2seq_sep_onlyk_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_divintact_base')
def dptree2seq_sep_onlyk_divintact_base(args):
    # args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    # args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    args.src_len_norm = getattr(args, 'src_len_norm', 'intact')
    dptree2seq_sep_onlyk_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_nodptcross_onlyk_base')
def dptree2seq_sep_nodptcross_onlyk_base(args):
    args.decoder_type = getattr(args, 'decoder_type', DPTree2SeqSeparateNoDPTCrossTransformerDecoder)
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_nodptcross_onlyk_divintact_base')
def dptree2seq_sep_nodptcross_onlyk_divintact_base(args):

    # args.decoder_type = getattr(args, 'decoder_type', DPTree2SeqSeparateNoDPTCrossTransformerDecoder)
    # args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    # args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    args.src_len_norm = getattr(args, 'src_len_norm', 'intact')
    dptree2seq_sep_nodptcross_onlyk_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_tiny')
def dptree2seq_sep_onlyk_tiny(args):
    # args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    # args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)
    dptree2seq_sep_onlyk_base(args)



@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_avg_onlyk_base')
def dptree2seq_sep_avg_onlyk_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateRootAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onseq_k_base')
def dptree2seq_sep_onseq_k_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualNoAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeOnSeqAttention)
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_avg_onseq_k_base')
def dptree2seq_sep_avg_onseq_k_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualRootAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeOnSeqAttention)
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onseq_k_crosstree_base')
def dptree2seq_sep_onseq_k_crosstree_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualNoAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeOnSeqCrossTreeAttention)
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_avg_onseq_k_crosstree_base')
def dptree2seq_sep_avg_onseq_k_crosstree_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualRootAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeOnSeqCrossTreeAttention)
    args.on_seq = getattr(args, 'on_seq', 'key')
    args.divide_src_len = getattr(args, 'divide_src_len', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_wsplit_base')
def dptree2seq_sep_onlyk_wsplit_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyWeightSplitAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_matsum_base')
def dptree2seq_sep_onlyk_matsum_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyMatSumAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_wsplit_matsum_base')
def dptree2seq_sep_onlyk_wsplit_matsum_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyWeightSplitMatSumAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq_sep', 'dptree2seq_sep_onlyk_rightup_base')
def dptree2seq_sep_onlyk_rightup_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeSeparateTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeSeparateOnlyKeyRightUpAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_indi_onlyk_base')
def dptree2seq_indi_onlyk_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeIndividualOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_indirnn_onlyk_base')
def dptree2seq_indirnn_onlyk_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeIndividualRNNOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_indirnn_avg_onlyk_base')
def dptree2seq_indirnn_avg_onlyk_base(args):
    args.encoder_type = getattr(args, 'encoder_type', DPTreeIndividualRootAverageTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', DPTreeIndividualRNNOnlyKeyAttention)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_tiny')
def dptree2seq_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 128)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 128)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 2)

    dptree2seq_base(args)


@register_model_architecture('dptree_encoder', 'dptree_encoder_tiny')
def dptree_encoder_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_base(args)


@register_model_architecture('dptree_encoder', 'dptree_encoder_onlyk_tiny')
def dptree_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_onlyk_base(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny')
def dptree_sep_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_base(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v2')
def dptree_sep_encoder_onlyk_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v3')
def dptree_sep_encoder_onlyk_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    # args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v3_ffn512')
def dptree_sep_encoder_onlyk_tiny_v3_ffn512(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v4')
def dptree_sep_encoder_onlyk_tiny_v4(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v5')
def dptree_sep_encoder_onlyk_tiny_v5(args):
    # args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 10)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onlyk_tiny_v6')
def dptree_sep_encoder_onlyk_tiny_v6(args):
    # args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
    dptree_sep_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_encoder', 'dptree2seq_sep_avg_onlyk_tiny')
def dptree2seq_sep_avg_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_avg_onlyk_base(args)


@register_model_architecture('dptree_encoder', 'dptree2seq_sep_avg_onlyk_tiny_v3')
def dptree2seq_sep_avg_onlyk_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree2seq_sep_avg_onlyk_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onseq_k_tiny')
def dptree_sep_encoder_onseq_k_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onseq_k_base(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onseq_k_tiny_v3')
def dptree_sep_encoder_onseq_k_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_encoder_onseq_k_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onseq_k_nodivide_tiny')
def dptree_sep_encoder_onseq_k_nodivide_tiny(args):
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree_sep_encoder_onseq_k_tiny(args)


@register_model_architecture('dptree_sep_encoder', 'dptree_sep_encoder_onseq_k_nodivide_tiny_v3')
def dptree_sep_encoder_onseq_k_nodivide_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_encoder_onseq_k_nodivide_tiny(args)



# - Node based ----------------------------------


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny')
def dptree_sep_node_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_v2')
def dptree_sep_node_encoder_onlyk_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_sep_node_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_v3')
def dptree_sep_node_encoder_onlyk_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_v4')
def dptree_sep_node_encoder_onlyk_tiny_v4(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_v5')
def dptree_sep_node_encoder_onlyk_tiny_v5(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 10)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_v6')
def dptree_sep_node_encoder_onlyk_tiny_v6(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 10)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_rightup_tiny')
def dptree_sep_node_encoder_onlyk_rightup_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_rightup_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_rightup_tiny_v2')
def dptree_sep_node_encoder_onlyk_rightup_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_sep_node_encoder_onlyk_rightup_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_rightup_tiny_v3')
def dptree_sep_node_encoder_onlyk_rightup_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_rightup_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_rightup_tiny_v4')
def dptree_sep_node_encoder_onlyk_rightup_tiny_v4(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_rightup_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_wsplit_tiny')
def dptree_sep_node_encoder_onlyk_wsplit_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_wsplit_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_wsplit_tiny_v2')
def dptree_sep_node_encoder_onlyk_wsplit_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_sep_node_encoder_onlyk_wsplit_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_wsplit_tiny_v3')
def dptree_sep_node_encoder_onlyk_wsplit_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_wsplit_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_wsplit_tiny_v4')
def dptree_sep_node_encoder_onlyk_wsplit_tiny_v4(args):
    # args.dropout = getattr(args, 'dropout', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.5)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.5)
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onlyk_wsplit_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_matsum_tiny')
def dptree_sep_node_encoder_onlyk_matsum_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_matsum_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_wsplit_matsum_tiny')
def dptree_sep_node_encoder_onlyk_wsplit_matsum_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onlyk_wsplit_matsum_base(args)

# ----------------------------------------------------------------------------------


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_avg_node_onlyk_tiny')
def dptree_sep_avg_node_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_avg_onlyk_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_avg_node_onlyk_tiny_v3')
def dptree_sep_avg_node_onlyk_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_avg_node_onlyk_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onseq_k_tiny')
def dptree_sep_node_encoder_onseq_k_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onseq_k_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onseq_k_tiny_v3')
def dptree_sep_node_encoder_onseq_k_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_encoder_onseq_k_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_avg_onseq_k_base_tiny')
def dptree_sep_node_avg_onseq_k_base_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_avg_onseq_k_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_avg_onseq_k_base_tiny_v3')
def dptree_sep_node_avg_onseq_k_base_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_node_avg_onseq_k_base_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onseq_k_nodivide_tiny')
def dptree_sep_node_encoder_onseq_k_nodivide_tiny(args):
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree_sep_node_encoder_onseq_k_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onseq_k_nodivide_tiny_v3')
def dptree_sep_node_encoder_onseq_k_nodivide_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree_sep_node_encoder_onseq_k_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_avg_onseq_k_base_nodivide_tiny')
def dptree_sep_node_avg_onseq_k_base_nodivide_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree2seq_sep_avg_onseq_k_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_avg_onseq_k_base_nodivide_tiny_v3')
def dptree_sep_node_avg_onseq_k_base_nodivide_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree_sep_node_avg_onseq_k_base_nodivide_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_onseq_k_crosstree_base_nodivide_tiny')
def dptree_sep_onseq_k_crosstree_base_nodivide_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree2seq_sep_onseq_k_crosstree_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_onseq_k_crosstree_base_nodivide_tiny_v3')
def dptree_sep_onseq_k_crosstree_base_nodivide_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.divide_src_len = getattr(args, 'divide_src_len', False)
    dptree_sep_onseq_k_crosstree_base_nodivide_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_onseq_k_crosstree_base_tiny')
def dptree_sep_onseq_k_crosstree_base_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_onseq_k_crosstree_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_onseq_k_crosstree_base_tiny_v3')
def dptree_sep_onseq_k_crosstree_base_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree_sep_onseq_k_crosstree_base_tiny(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree2seq_sep_avg_onseq_k_crosstree_tiny')
def dptree2seq_sep_avg_onseq_k_crosstree_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_sep_avg_onseq_k_crosstree_base(args)


@register_model_architecture('dptree_sep_node_encoder', 'dptree2seq_sep_avg_onseq_k_crosstree_tiny_v3')
def dptree2seq_sep_avg_onseq_k_crosstree_tiny_v3(args):
    args.dropout = getattr(args, 'dropout', 0.5)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    dptree2seq_sep_avg_onseq_k_crosstree_tiny(args)





# todo:--------------------------------


@register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny')
def dptree_indi_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_indi_onlyk_base(args)


@register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny_v2')
def dptree_indi_encoder_onlyk_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_indi_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_indi_encoder', 'dptree_indirnn_encoder_onlyk_tiny')
def dptree_indirnn_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_indirnn_onlyk_base(args)


@register_model_architecture('dptree_indi_encoder', 'dptree_indirnn_encoder_onlyk_tiny_v2')
def dptree_indirnn_encoder_onlyk_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_indirnn_encoder_onlyk_tiny(args)


@register_model_architecture('dptree_indi_encoder', 'dptree_indirnn_avg_encoder_onlyk_tiny')
def dptree_indirnn_avg_encoder_onlyk_tiny(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    dptree2seq_indirnn_avg_onlyk_base(args)


@register_model_architecture('dptree_indi_encoder', 'dptree_indirnn_avg_encoder_onlyk_tiny_v2')
def dptree_indirnn_avg_encoder_onlyk_tiny_v2(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    dptree_indirnn_avg_encoder_onlyk_tiny(args)



# @register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny_v3')
# def dptree_indi_encoder_onlyk_tiny_v3(args):
#     args.dropout = getattr(args, 'dropout', 0.5)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     dptree_indi_encoder_onlyk_tiny(args)
#
#
# @register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny_v4')
# def dptree_indi_encoder_onlyk_tiny_v4(args):
#     args.dropout = getattr(args, 'dropout', 0.5)
#     args.encoder_layers = getattr(args, 'encoder_layers', 3)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
#     dptree_indi_encoder_onlyk_tiny(args)
#
#
# @register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny_v5')
# def dptree_indi_encoder_onlyk_tiny_v5(args):
#     # args.dropout = getattr(args, 'dropout', 0.5)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 10)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
#     dptree_indi_encoder_onlyk_tiny(args)
#
#
# @register_model_architecture('dptree_indi_encoder', 'dptree_indi_encoder_onlyk_tiny_v6')
# def dptree_indi_encoder_onlyk_tiny_v6(args):
#     # args.dropout = getattr(args, 'dropout', 0.5)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 300)
#     dptree_indi_encoder_onlyk_tiny(args)






# @register_model_architecture('dptree_sep_node_encoder', 'dptree_sep_node_encoder_onlyk_tiny_binary')
# def dptree_sep_node_encoder_onlyk_tiny_binary(args):
#     args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
#     args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
#     args.encoder_layers = getattr(args, 'encoder_layers', 2)
#     args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
#     args.convert_to_binary = getattr(args, 'convert_to_binary', True)
#     dptree2seq_sep_onlyk_base(args)


@register_model_architecture('dptree_sep_nli_encoder', 'dptree_sep_nli_encoder_onlyk_base')
def dptree_sep_nli_encoder_onlyk_base(args):
    dptree2seq_sep_onlyk_base(args)


@register_model_architecture('dptree_sep_nli_concat_encoder', 'dptree_sep_nli_concat_encoder_onlyk_base')
def dptree_sep_nli_concat_encoder_onlyk_base(args):
    dptree2seq_sep_onlyk_base(args)





@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de')
def dptree2seq_base_wmt_en_de(args):
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_onlyk_wmt_en_de')
def dptree2seq_onlyk_wmt_en_de(args):
    dptree2seq_onlyk_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_t2t')
def dptree2seq_base_wmt_en_de_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_vas_big')
def dptree2seq_big_wmt_en_de_vas(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    dptree2seq_base(args)


@register_model_architecture('dptree2seq', 'dptree2seq_wmt_en_de_t2t_big')
def dptree2seq_big_wmt_en_de_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)

    dptree2seq_big_wmt_en_de_vas(args)
