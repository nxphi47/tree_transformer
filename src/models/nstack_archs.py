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

from fairseq.models.transformer import *

from .nstack_transformer import *





# args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', True)
#
# args.wnstack_norm = getattr(args, 'wnstack_norm', 'none')
# args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.DEFAULT)
# args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
# args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
# args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
# args.node_embed_init = getattr(args, 'node_embed_init', 'embed')


# distinct weights classification
# # weighted-on-aff
#
# dwnstack_node_class_onaff_tiny_sqrtmean_v3_ffn512
# dwnstack_node_class_onaff_tiny_mean_v3_ffn512


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_v3_ffn512')
def dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'embed')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# FIXME: hier_embedding NO work OnAffinity
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_hier_v3_ffn512')
def dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_hier_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'embed')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512')
def dwnstack_node_class_onaff_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512')
# def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512(args):
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
#     args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
#     args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
#     args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
#     args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
#     args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
#
#     args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
#     dwnstack_node_class_onaff_tiny_v3_ffn512(args)



@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base')
def dwnstack2seq_node_onaffvalue_base(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_sqrtmean')
def dwnstack2seq_node_onaffvalue_base_sqrtmean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    # args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upmean_mean')
def dwnstack2seq_node_onaffvalue_base_upmean_mean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upsqrtmean_sqrtmean')
def dwnstack2seq_node_onaffvalue_base_upsqrtmean_sqrtmean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross')
def dwnstack2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross')
def dwnstack2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_sqrtmean_mlesub')
def dwnstack2seq_node_onaffvalue_base_sqrtmean_mlesub(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_sqrtmean_mlesub_hier')
def dwnstack2seq_node_onaffvalue_base_sqrtmean_mlesub_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upmean_mean_hier')
def dwnstack2seq_node_onaffvalue_base_upmean_mean_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)




# TODO: IWSLT'14
@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    add_iwslt(args)
    nstack2seq_base(args)

@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_dsigmoid')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_dsigmoid(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'sigmoid')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_dtanh')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_dtanh(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'tanh')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean')
def dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_hier')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_hier')
def dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross')
def dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier')
def dwnstack2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hiershare')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hiershare(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', True)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_nocross')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_nocross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_cross = getattr(args, 'nstack_cross', False)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaff_base_upmean_mean_nocross')
def dwnstack2seq_node_iwslt_onaff_base_upmean_mean_nocross(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_cross = getattr(args, 'nstack_cross', False)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack2seq', 'dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_hiershare')
def dwnstack2seq_node_iwslt_onaffvalue_base_upmean_mean_hiershare(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', True)
    add_iwslt(args)
    nstack2seq_base(args)


# fixme: ---- nstack merge ------
@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hier')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hiershare')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hiershare(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hiershare')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hiershare(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_share = getattr(args, 'nstack_hier_embed_share', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_sqrtmean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_sqrtmean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc35_allcross')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc35_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [False, False, False, True, True, True])
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc15_allcross')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc15_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [False, True, True, True, True, True])
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc25_allcross')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc25_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [False, False, True, True, True, True])
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [True, True, True, False, False, False])
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross_wtanh')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross_wtanh(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [True, True, True, False, False, False])
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'tanh')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross_wsigmoid')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc02_allcross_wsigmoid(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [True, True, True, False, False, False])
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'sigmoid')
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hierright')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_hierright(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.hier_embed_right = getattr(args, 'hier_embed_right', False)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hierfull')
def dwnstack_merge2seq_node_iwslt_onaffvalue_base_upmean_mean_mlesubenc_allcross_hierfull(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


# NMT: on value, on key
@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    # add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hierfull')
def dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hierfull(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onkey_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onkey_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    # add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onkeyvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_iwslt_onkeyvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnKeyValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    # add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_iwslt_onkeyvalue_base_upmean_mean_mlesubenc_allcross_hierfull')
def dwnstack_merge2seq_node_iwslt_onkeyvalue_base_upmean_mean_mlesubenc_allcross_hierfull(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnKeyValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_hier_full_dim(args)
    add_iwslt(args)
    nstack2seq_base(args)








# --- todo: for WMT"16 base
@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_hier')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc02_allcross')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc02_allcross(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [True, True, True, False, False, False])
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hierfull')
def dwnstack_merge2seq_node_onaffvalue_base_upsqrtmean_sqrtmean_mlesubenc_allcross_hierfull(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_hier_full_dim(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hierfull')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hierfull(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 200)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_hier_full_dim(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_big_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_big_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_vaswani_big(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_big_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_big_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 200)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_vaswani_big(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_supertiny_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier')
def dwnstack_supertiny_merge2seq_node_onvalue_base_upmean_mean_mlesubenc_allcross_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 200)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_s2s_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_big_merge2seq_node_onaffvalue_base_upmean_mean_hier')
def dwnstack_big_merge2seq_node_onaffvalue_base_upmean_mean_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_vaswani_big(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge2seq', 'dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross02_hier')
def dwnstack_merge2seq_node_onaffvalue_base_upmean_mean_mlesubenc_allcross02_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.decoder_type = getattr(args, 'decoder_type', NstackMerge2SeqTransformerDecoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cross_nstack_mask_fn = getattr(args, 'cross_nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_df_layer = getattr(args, 'nstack_mask_df_layer', [True, True, True, False, False, False])
    nstack2seq_base(args)



# ----------------------------------------------------------------------------------------------------------------------


# fixme: nstack merge for classification
@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onvalue_base_upmean_mean_mlesub_hier')
def dwnstack_merge_tiny_onvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tinyd128f256_onvalue_base_upmean_mean_mlesub_hier')
def dwnstack_merge_tinyd128f256_onvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny_d128f256(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tinyd128f512_onvalue_base_upmean_mean_mlesub_hier_1nonodes')
def dwnstack_merge_tinyd128f512_onvalue_base_upmean_mean_mlesub_hier_1nonodes(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 512)
    add_tiny_d128f256(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onkeyvalue_base_upmean_mean_mlesub_hier')
def dwnstack_merge_tiny_onkeyvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnKeyValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_lm_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesublm_hier')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesublm_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE_LM)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_3v512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier_3v512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_3v512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_3v512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_1nonodes_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_1nonodes_3v512(args):
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_pretrainedlinear_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_pretrainedlinear_3v512(args):
    args.pretrained_linear = getattr(args, 'pretrained_linear', False)
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args)


@register_model_architecture('nstack_merge_lm_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesublm_hier_3v512')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesublm_hier_3v512(args):
    add_tiny_v3_ffn512(args)
    dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesublm_hier(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_7v256')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_7v256(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny_v7h4ffn256(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_encoder', 'dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_7v256_embedxl')
def dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier_7v256_embedxl(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 2000)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 150)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', MergeWeightMask.LEAVES_SUBTREE)
    add_tiny_v7h4ffn256(args)
    nstack2seq_base(args)




# TODO: relateness

@register_model_architecture('nstack_merge_relateness_encoder', 'relate_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier')
def relate_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


# TODO: relateness concat
@register_model_architecture('nstack_merge_relateness_concat_encoder', 'relateconcat_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier')
def relateconcat_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_mlesub_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    add_tiny(args)
    nstack2seq_base(args)


@register_model_architecture('nstack_merge_relateness_concat_encoder', 'relateconcat_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier')
def relateconcat_dwnstack_merge_tiny_onaffvalue_base_upmean_mean_hier(args):
    args.encoder_type = getattr(args, 'encoder_type', NstackMergeTransformerEncoder)
    args.dptree_class = getattr(args, 'dptree_class', MergeStackNodesOnAffinityValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 50)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    add_tiny(args)
    nstack2seq_base(args)






# ---------------------------------------------------------------------------------

@register_model_architecture('nstack2seq', 'dwnstack2seq_node_onvalue_base_sqrtmean')
def dwnstack2seq_node_onvalue_base_sqrtmean(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    nstack2seq_base(args)



# on-aff-value
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_v3_ffn512(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# don't work:: attn = torch.bmm(attn_weights, values) --> Nan, weight: [0, 1.0]
# and multi-loss explode
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)

# dwnstack_node_class_onaff_tiny()


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier')
def dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier')
def dwnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onquery_tiny_upmean_sqrtmean_mlesub_hier')
def dwnstack_node_class_onquery_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnQueryAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onquery_tiny_upmean_sqrtmean_mlesub_hier_v3_ffn512')
def dwnstack_node_class_onquery_tiny_upmean_sqrtmean_mlesub_hier_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnQueryAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)



@register_model_architecture('nstack_node_encoder', 'csnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier')
def csnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackCSumOnValueAttention)

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'csnstack_node_class_onkey_tiny_upmean_sqrtmean_mlesub_hier')
def csnstack_node_class_onkey_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackCSumOnKeyAttention)

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'csnstack_node_class_onkeyvalue_tiny_upmean_sqrtmean_mlesub_hier')
def csnstack_node_class_onkeyvalue_tiny_upmean_sqrtmean_mlesub_hier(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.cum_node = getattr(args, 'cum_node', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackCSumOnKeyValueAttention)

    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)



@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier_1nonodes')
def dwnstack_node_class_onvalue_tiny_upmean_sqrtmean_mlesub_hier_1nonodes(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny7_upmean_sqrtmean_mlesub_hier_1nonodes')
def dwnstack_node_class_onvalue_tiny7_upmean_sqrtmean_mlesub_hier_1nonodes(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    add_tiny_v7(args)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onkey_tiny7_upmean_sqrtmean_mlesub_hier_1nonodes')
def dwnstack_node_class_onkey_tiny7_upmean_sqrtmean_mlesub_hier_1nonodes(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnKeyAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    add_tiny_v7(args)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier_1nonodes')
def dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier_1nonodes(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny(args)



@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_hier_v3_ffn512(args):
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


@register_model_architecture('nstack_nli_encoder', 'dwnstack_node_nli_concat_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512')
def dwnstack_node_nli_concat_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args):
    args.concat = getattr(args, 'concat', True)
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


@register_model_architecture('nstack_nli_encoder', 'dwnstack_node_nli_concat_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_v3_ffn512')
def dwnstack_node_nli_concat_class_onaffvalue_tiny_upmean_sqrtmean_mlesub_v3_ffn512(args):
    # FIXME: this mlesub is wrong!, it was designed for ALL_ALL
    args.concat = getattr(args, 'concat', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


@register_model_architecture('nstack_nli_encoder', 'dwnstack_node_nli_concat_class_onaffvalue_tiny_upsqrtmean_sqrtmean_mlesub_v3_ffn512')
def dwnstack_node_nli_concat_class_onaffvalue_tiny_upsqrtmean_sqrtmean_mlesub_v3_ffn512(args):
    # FIXME: this mlesub is wrong!, it was designed for ALL_ALL
    args.concat = getattr(args, 'concat', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


@register_model_architecture('nstack_nli_encoder', 'dwnstack_node_nli_concat_class_onaffvalue_tiny_upmean_sqrtmean_all_v3_ffn512')
def dwnstack_node_nli_concat_class_onaffvalue_tiny_upmean_sqrtmean_all_v3_ffn512(args):
    args.concat = getattr(args, 'concat', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


@register_model_architecture('nstack_nli_encoder', 'dwnstack_node_nli_concat_class_onaffvalue_tiny_upsqrtmean_sqrtmean_all_v3_ffn512')
def dwnstack_node_nli_concat_class_onaffvalue_tiny_upsqrtmean_sqrtmean_all_v3_ffn512(args):
    args.concat = getattr(args, 'concat', True)
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v3_ffn512(args)


# fixme: results glove-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


# fixme: results glove-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)



@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_v4_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_v4_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstacck_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


# fixme: results glove-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v4_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    dwnstack_node_class_onaff_tiny_v5(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstacck_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    dwnstack_node_class_onaff_tiny_v5(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_dwsigmoid_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_dwsigmoid_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstacck_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'sigmoid')
    dwnstack_node_class_onaff_tiny_v5(args)


# fixme: results random-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_dwtanh_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_dwtanh_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstacck_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.dwstack_proj_act = getattr(args, 'dwstack_proj_act', 'tanh')
    dwnstack_node_class_onaff_tiny_v5(args)


# fixme: results glove-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_node0_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v5(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mansub5_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mansub5_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.MUTUALANCESTORS_SUBTREE)
    args.mutual_ancestor_level = getattr(args, 'mutual_ancestor_level', 5)

    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)



# todo: for ablation studies
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_mean_upmean_mlesub_hier_v3_ffn512_ab')
def dwnstack_node_class_onaffvalue_tiny_mean_upmean_mlesub_hier_v3_ffn512_ab(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_mean_upmean_mlesub_v3_ffn512_ab')
def dwnstack_node_class_onaffvalue_tiny_mean_upmean_mlesub_v3_ffn512_ab(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', False)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 100)
    # args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_mean_upmean_hier_v3_ffn512_ab')
def dwnstack_node_class_onaffvalue_tiny_mean_upmean_hier_v3_ffn512_ab(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.ALL_ALL)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 100)
    # args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)








@register_model_architecture('nstack_node_encoder', 'swdwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512')
def swdwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackSepWDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'swdwnstack_node_class_onaff_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512')
def swdwnstack_node_class_onaff_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackSepWDistinctWeightedOnAffinityAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)



@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_transit_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_transit_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.transition_act = getattr(args, 'transition_act', 'none')
    args.transition_dropout = getattr(args, 'transition_dropout', 0.4)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackConvDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upsqrtmean_mlesub_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upsqrtmean_mlesub_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila1_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila1_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_splitupdown_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila1_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_splitupdown_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila1_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedSplitUpDownOnAffinityValueAttention)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_splitupdown_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila12_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_splitupdown_tiny_sqrtmean_upsqrtmean_mlesub_hier_vanila12_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedSplitUpDownOnAffinityValueAttention)

    args.vanilla_layers = getattr(args, 'vanilla_layers', 1)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVESANCESTORS_SUBTREELEAVES)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mleasubl_hier_1nonodes_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mleasubl_hier_1nonodes_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVESANCESTORS_SUBTREELEAVES)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_upmean_mlesub_hier_1nonodes_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.wnstack_up_norm = getattr(args, 'wnstack_up_norm', 'mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v4_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVESANCESTORS_SUBTREELEAVES)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)




@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v4_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v4_ffn512')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mleasubl_hier_1nonodes_v4_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVESANCESTORS_SUBTREELEAVES)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)

    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v4_ffn512(args)


# fixme: results glove-init good
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v5')
def dwnstack_node_class_onvalue_tiny_sqrtmean_mlesub_hier_1nonodes_v5(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    # args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    args.first_layer_nonodes = getattr(args, 'first_layer_nonodes', True)
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnValueAttention)
    dwnstack_node_class_onaff_tiny_v5(args)



# embed_pretrained_no_scale


# fixme: glove: not as good as aboe: 46 - 84
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_noscale_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_noscale_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', True)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# fixme: glove:
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_noscale_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_noscale_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', True)
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)


# fixme: glove:
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_noscale_v3_ffn512')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_noscale_v3_ffn512(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
    args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
    args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
    args.embed_pretrained_no_scale = getattr(args, 'embed_pretrained_no_scale', True)
    args.node_embed_init = getattr(args, 'node_embed_init', 'zero')
    dwnstack_node_class_onaff_tiny_v3_ffn512(args)




# on-aff-value
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny')
def dwnstack_node_class_onaffvalue_tiny(args):
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny(args)


# don't work:: attn = torch.bmm(attn_weights, values) --> Nan, weight: [0, 1.0]
# and multi-loss explode
@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    dwnstack_node_class_onaff_tiny(args)


@register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub')
def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub(args):
    args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
    args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
    args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
    dwnstack_node_class_onaff_tiny(args)

#
# # fixme: results random-init good
# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier')
# def dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier(args):
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'sqrt_mean')
#     args.dptree_class = getattr(args, 'dptree_class', NodeStackDistinctWeightedOnAffinityValueAttention)
#     args.nstack_mask_fn = getattr(args, 'nstack_mask_fn', WeightMask.LEAVES_SUBTREE)
#     args.nstack_hier_embed = getattr(args, 'nstack_hier_embed', True)
#     args.nstack_hier_embed_max_horiz = getattr(args, 'nstack_hier_embed_max_horiz', 100)
#     args.nstack_hier_embed_max_ver = getattr(args, 'nstack_hier_embed_max_ver', 1024)
#     dwnstack_node_class_onaff_tiny(args)
#

"""
dwnstack_node_class_onaffvalue_tiny_sqrtmean_mlesub_hier_node0_v3_ffn512
| epoch 012 | valid on 'valid' subset | loss 50.769 | ppl 1919045585894984.75 | num_updates 1048 | best_loss 48.091 | acc 0.54627 | nn_nsents 84.6923 | bin_acc_sum 61.5385 | bin_acc_avg 0.74008 | bin_acc 76.1535 | bin_acc_av 0.726612 | all_acc 0.525886
| epoch 012 | valid on 'valid1' subset | loss 50.801 | ppl 1962237053559581.50 | num_updates 1048 | best_loss 48.091 | acc 0.541848 | nn_nsents 92.0833 | bin_acc_sum 67.8333 | bin_acc_avg 0.746676 | bin_acc 84.2299 | bin_acc_av 0.736652 | all_acc 0.525792



| epoch 001 | valid on 'valid' subset | loss 87.141 | ppl 1705 | num_updates 88 | nn_nsents 0.792007 | bin_acc_avg 0.396912 | bin_acc 0.501147 | bin_acc_av 0.501147 | all_acc 0.206176 | all_target 2.03815 | all_bin_target 0.509174
| epoch 001 | valid on 'valid1' subset | loss 86.537 | ppl 112 | num_updates 88 | nn_nsents 0.823982 | bin_acc_avg 0.404525 | bin_acc 0.490939 | bin_acc_av 0.490939 | all_acc 0.178281 | all_target 2.05294 | all_bin_target 0.499176

| epoch 001 | valid on 'valid' subset | loss 68.323 | ppl 3692 | num_updates 80 | nn_nsents 84.6923 | bin_acc_avg 0.34723 | bin_acc 43.7411 | bin_acc_av 0.378747 | all_acc 0.175295    | all_target 2.23161 | all_bin_target 0.647593
| epoch 001 | valid on 'valid1' subset | loss 68.398 | ppl 388 | num_updates 80 | nn_nsents 92.0833 | bin_acc_avg 0.375478 | bin_acc 49.1421 | bin_acc_av 0.395475 | all_acc 0.183258   | all_target 2.24751 | all_bin_target 0.647059
"""






# @register_model_architecture('nstack_node_encoder', 'dwnstack_node_class_onaff_tiny_mean_v3_ffn512')
# def dwnstack_node_class_onaff_tiny_mean_v3_ffn512(args):
#     # args.wnstack_include_leaves = getattr(args, 'wnstack_include_leaves', False)
#     args.wnstack_norm = getattr(args, 'wnstack_norm', 'mean')
#     dwnstack_node_class_onaff_tiny_v3_ffn512(args)











