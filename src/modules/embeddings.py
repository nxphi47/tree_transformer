import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
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

from fairseq.models import transformer


def OnesPlaceholderEmbedding(num_embeddings, embedding_dim, padding_idx, placeholder_idx=4):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    nn.init.constant_(m.weight[placeholder_idx], 1)
    return m


def PretrainedEmbedding(num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path):
    print(f'| Load embedding at {pretrain_path}')
    emb = transformer.Embedding(num_embeddings, embedding_dim, padding_idx)
    embed_dict = utils.parse_embedding(pretrain_path)

    mask_factor = []
    for idx in range(len(dictionary)):
        token = dictionary[idx]
        if token in embed_dict:
            emb.weight.data[idx] = embed_dict[token]
            mask_factor.append(0)
        else:
            mask_factor.append(1)

    mask_factor = torch.tensor(mask_factor, dtype=torch.float32).unsqueeze_(-1).expand(len(mask_factor), embedding_dim)

    def hook(grad):
        grad *= mask_factor.type_as(grad)
        return grad

    emb.weight.register_hook(hook)
    return emb


def FreePretrainedEmbedding(num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path):
    print(f'| Load FreePretained embedding at {pretrain_path}')
    emb = transformer.Embedding(num_embeddings, embedding_dim, padding_idx)
    embed_dict = utils.parse_embedding(pretrain_path)

    # mask_factor = []
    for idx in range(len(dictionary)):
        token = dictionary[idx]
        if token in embed_dict:
            emb.weight.data[idx] = embed_dict[token]
            # mask_factor.append(0)
        # else:
        # mask_factor.append(1)

    special_toks = ['-LRB-', '-RRB-']
    bind_toks = ['(', ')']
    for tok, btok in zip(special_toks, bind_toks):
        idx = dictionary.index(tok)
        if btok in embed_dict and idx != dictionary.unk_index:
            print(f'Replace embed for {tok} {btok} - {idx}')
            emb.weight.data[idx] = embed_dict[btok]

    return emb


class LinearPretrainedEmbedding(nn.Module):

    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path):
        super().__init__()
        self.args = args
        self.dropout = getattr(args, 'dropout', 0.0)
        self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.dictionary = dictionary
        self.pretrain_path = pretrain_path

        self.embedding = PretrainedEmbedding(num_embeddings, self.pretrain_dim, padding_idx, dictionary, pretrain_path)
        self.linear = transformer.Linear(self.pretrain_dim, embedding_dim, bias=False)
        self.layer = nn.Sequential(
            self.embedding, self.dropout_layer, self.linear
        )

    def forward(self, inputs):
        return self.layer(inputs)


class FinetunePretrainedEmbedding(nn.Module):

    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path):
        super().__init__()
        self.args = args
        self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
        self.tune_epoch = getattr(args, 'tune_epoch', 10000000)
        self.current_epoch = 0
        self.finetuning = False
        self.flip_switch = True

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.dictionary = dictionary
        self.pretrain_path = pretrain_path

        self.embedding = self.build_embed(
            num_embeddings, self.pretrain_dim, padding_idx, dictionary, pretrain_path, self.tune_epoch)

    def extra_repr(self):
        return f'pretrain_dim={self.pretrain_dim},tune_epoch={self.tune_epoch}'

    def build_embed(self, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, tune_epoch):
        print(f'| Load embedding at {pretrain_path}')
        emb = transformer.Embedding(num_embeddings, embedding_dim, padding_idx)
        embed_dict = utils.parse_embedding(pretrain_path)

        mask_factor = []
        for idx in range(len(dictionary)):
            token = dictionary[idx]
            if token in embed_dict:
                emb.weight.data[idx] = embed_dict[token]
                mask_factor.append(0)
            else:
                mask_factor.append(1)

        mask_factor = torch.tensor(mask_factor, dtype=torch.float32).unsqueeze_(-1).expand(
            len(mask_factor), embedding_dim)

        def hook(grad):
            if self.flip_switch:
                print(f'Flip switch: {self.finetuning}')
                self.flip_switch = False
            if not self.finetuning:
                grad *= mask_factor.type_as(grad)
            return grad

        emb.weight.register_hook(hook)
        return emb

    def turn_finetune(self, on=True):
        self.finetuning = on
        mess = "ON" if self.finetuning else "OFF"
        print(f'Turn {mess} finetuning')
        self.flip_switch = True

    def forward(self, x):
        return self.embedding(x)


class SeparatePretrainedEmbedding(nn.Module):
    def __init__(
            self, args, num_embeddings, embedding_dim, padding_idx, dictionary,
            pretrain_path, freeze=True, enforce_unk=False, same_dist=False
    ):
        super().__init__()
        self.args = args
        self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
        self.tune_epoch = getattr(args, 'tune_epoch', 10000000)
        self.current_epoch = 0
        self.finetuning = False
        self.flip_switch = True

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.dictionary = dictionary
        self.pretrain_path = pretrain_path
        self.freeze = freeze
        self.enforce_unk = enforce_unk
        self.same_dist = same_dist

        self.reordering, self.pretrained_emb, self.new_embed = self.build_embed(
            num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, self.tune_epoch
        )

        if self.pretrained_pad_idx is not None:
            self.final_pad_idx = self.pretrained_pad_idx
        else:
            self.final_pad_idx = self.pretrained_emb.weight.size(0) + self.new_pad_idx

        if self.pretraied_unk_idx is not None:
            self.final_unk_idx = self.pretraied_unk_idx
        else:
            self.final_unk_idx = self.pretrained_emb.weight.size(0) + self.new_unk_idx

    def create_new_embed(self, num_embeds, embedding_dim, pad_idx, pretrained_emb=None):
        if self.same_dist:
            _mean = pretrained_emb.weight.mean()
            _std = pretrained_emb.weight.std()
            new_embed = nn.Embedding(num_embeds, embedding_dim, padding_idx=pad_idx)
            nn.init.normal_(new_embed.weight, mean=_std, std=_std)
            nn.init.constant_(new_embed.weight[pad_idx], 0)
        else:
            new_embed = transformer.Embedding(num_embeds, embedding_dim, pad_idx)
        return new_embed

    def build_embed(self, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, tune_epoch):
        print(f'| Load embedding dict at {pretrain_path}')

        pretrained_embed_dict = utils.parse_embedding(pretrain_path)

        remap_pretrained_embed = []

        remap_pretrained_indices = []
        remap_new_indices = []

        self.pretrained_pad_idx = None
        self.new_pad_idx = None
        self.pretraied_unk_idx = None
        self.new_unk_idx = None

        unk_idx = dictionary.unk_index

        phrase_label_idx = []
        nonphrase_label_idx = []
        for idx in range(len(dictionary)):
            token = dictionary[idx]
            is_padding = idx == padding_idx
            is_unk = idx == unk_idx
            if token in pretrained_embed_dict:
                remap_pretrained_embed.append(pretrained_embed_dict[token])
                remap_pretrained_indices.append(idx)
                if is_padding:
                    print(f'Padding {idx} in pretained embeddings, map to {len(remap_pretrained_indices) - 1}')
                    self.pretrained_pad_idx = len(remap_pretrained_indices) - 1
                if is_unk:
                    print(f'UNK {idx} in pretained embeddings, map to {len(remap_pretrained_indices) - 1}')
                    self.pretraied_unk_idx = len(remap_pretrained_indices) - 1
            else:
                remap_new_indices.append(idx)
                if "_label" in token:
                    phrase_label_idx.append(idx)
                else:
                    nonphrase_label_idx.append(idx)

                if is_padding:
                    print(f'Padding {idx} in new embeddings, map to {len(remap_new_indices) - 1}')
                    self.new_pad_idx = len(remap_new_indices) - 1
                if is_unk:
                    print(f'UNK {idx} in new embeddings, map to {len(remap_new_indices) - 1}')
                    self.new_unk_idx = len(remap_new_indices) - 1

        assert self.pretrained_pad_idx is None or self.new_pad_idx is None
        assert self.pretraied_unk_idx is None or self.new_unk_idx is None
        if self.pretraied_unk_idx is not None:
            self.final_unk_idx = self.pretraied_unk_idx
        else:
            self.final_unk_idx = len(remap_pretrained_embed) + self.new_unk_idx

        final_indices = remap_pretrained_indices + remap_new_indices

        reordering = sorted(range(len(final_indices)), key=lambda k: final_indices[k])

        if self.enforce_unk:
            print(f'Reordering indices for UNK, non-phrase={len(nonphrase_label_idx)}')
            if len(phrase_label_idx) == 0:
                raise ValueError(f'enforce unk, phrase-label empty, enforce will cause all phrase-lavel to UNK')
            redirect_reordering = []
            for idx in reordering:
                if idx in nonphrase_label_idx:
                    redirect_reordering.append(self.final_unk_idx)
                else:
                    redirect_reordering.append(idx)
            reordering = redirect_reordering

        reordering = nn.Parameter(torch.tensor(reordering), requires_grad=False)

        pretrained_embed = nn.Embedding.from_pretrained(
            torch.stack(remap_pretrained_embed), freeze=self.freeze, padding_idx=self.pretrained_pad_idx)
        # new_embed = transformer.Embedding(len(remap_new_indices), embedding_dim, self.new_pad_idx)
        new_embed = self.create_new_embed(len(remap_new_indices), embedding_dim, self.new_pad_idx, pretrained_emb=pretrained_embed)
        # [0, 1, 2, 3, 4]
        # [2, 3, 1, 0, 4]
        # [3, 2, 0, 1, 4]
        # 3 -> 1
        print(f'Phrase-label : {len(phrase_label_idx)}')
        if len(phrase_label_idx) == 0:
            print(f'WARNING !!! Phrase-label set empty!')
        return reordering, pretrained_embed, new_embed

    def switch_require_grads(self, on=True):
        print(f'Turn on embedding grads {on}')
        self.pretrained_emb.weight.requires_grad_(on)

    def turn_finetune(self, on=True):
        if self.finetuning != on:
            self.finetuning = on
            mess = "ON" if self.finetuning else "OFF"
            print(f'Turn {mess} finetuning')
            self.flip_switch = True
            self.switch_require_grads(on)

    def extra_repr(self):
        return f'pretrain_dim={self.pretrain_dim},tune_epoch={self.tune_epoch},new_pad={self.final_pad_idx},eunk={self.enforce_unk},sdist={self.same_dist}'

    def forward(self, x):

        reordered = F.embedding(x, self.reordering)
        total_embed = torch.cat([self.pretrained_emb.weight, self.new_embed.weight], 0)
        embed = F.embedding(
            reordered, total_embed, self.final_pad_idx,
            # self.new_embed.max_norm, self.new_embed.norm_type, self.new_embed.scale_grad_by_freq
        )
        return embed


class FinetuneSeparatePretrainedEmbedding(SeparatePretrainedEmbedding):

    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze=False,
                 enforce_unk=False, same_dist=True):
        assert not freeze
        super().__init__(
            args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze, enforce_unk, same_dist)

        def finetune_hook(grad):
            if self.flip_switch:
                print(f'Flip switch: {self.finetuning}')
                self.flip_switch = False
            if not self.finetuning:
                grad *= 0.0
            return grad

        self.pretrained_emb.weight.register_hook(finetune_hook)


class DownscaleSeparatePretrainedEmbedding(SeparatePretrainedEmbedding):

    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze=False,
                 enforce_unk=False, same_dist=True):
        super().__init__(args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze,
                         enforce_unk, same_dist)
        self.downscale_emb = getattr(args, 'downscale_emb', 0.1)

        def downscale_hook(grad):
            grad *= self.downscale_emb
            return grad

        self.new_embed.weight.register_hook(downscale_hook)

    def extra_repr(self):
        return f'{super().extra_repr()},downscale_emb={self.downscale_emb}'


class DownscaleFinetuneSeparatePretrainedEmbedding(FinetuneSeparatePretrainedEmbedding):

    def __init__(
            self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze=False,
            enforce_unk=False, same_dist=False, downscale_new=False):
        super().__init__(args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze,
                         enforce_unk, same_dist)
        self.downscale_emb = getattr(args, 'downscale_emb', 0.1)
        self.downscale_new = downscale_new

        def downscale_hook(grad):
            grad *= self.downscale_emb
            return grad

        if self.downscale_new:
            self.new_embed.weight.register_hook(downscale_hook)
        self.pretrained_emb.weight.register_hook(downscale_hook)

    def extra_repr(self):
        return f'{super().extra_repr()},downscale_emb={self.downscale_emb},down_new={self.downscale_new}'


def remap(dictionary, pre_dict, pad_idx=1):
    prev_indices = []
    new_indices = []
    prev_words = []
    new_words = []
    prev_pad = new_pad = None
    pad_token = dictionary[pad_idx]
    for idx in range(len(dictionary)):
        token = dictionary[idx]
        is_padding = idx == pad_idx
        if token in pre_dict:
            prev_indices.append(idx)
            prev_words.append(token)
            # if is_padding:
            #     prev_pad = len(prev_indices)
        else:
            new_indices.append(idx)
            new_words.append(token)
            # if is_padding:
            #     new_pad = len(new_indices)
    if pad_token in prev_words:
        prev_pad = prev_words.index(pad_token)
    else:
        assert pad_token in new_words
        new_pad = new_words.index(pad_token)
    final_indices = prev_indices + prev_indices
    reordering = sorted(range(len(final_indices)), key=lambda k: final_indices[k])
    return prev_words, new_words, reordering, prev_pad, new_pad


def run():
    # train.tok.clean.bpe.32000.p0.benepar_en2_large.v2.ende.bpe.w.de
    parts = ['p0','p1']
    langs = ['en', 'de']
    informat = 'train.tok.clean.bpe.32000.{}.benepar_en2_large.v2.ende.bpe.w.{}'
    outformat = 'train.tok.clean.bpe.32000.benepar_en2_large.v2.ende.bpe.w.{}'
    for l in langs:
        print(f'Process lang {l}')
        data = []
        for p in parts:
            file = informat.format(p, l)
            print(f'Open file {file}')
            with open(file, 'r') as f:
                lines = f.read().strip().split('\n')
                data.extend(lines)
                print(f'Data size: {len(data)}')
        outfile = outformat.format(l)
        print(f'Write: {outfile}')
        with open(outfile, 'w') as f:
            f.write('\n'.join(data))

#
# _dict = ['<>', '<pad>', 'xin', 'chao', 'toi', 'la', 'ha', 'uyen', 'rat', 'vui', 'gap', 'ban']
# _prev = ['ha', 'la', 'uyen', 'vui']
# print(len(_dict))
# print(list(range(len(_dict))))
# print(_dict)
#
# prev_words, new_words, reordering, prev_pad, new_pad = remap(_dict, _prev, pad_idx=1)
#

class PhraseAveragePretrainedEmbedding(nn.Module):
    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path):
        super().__init__()
        self.args = args
        self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
        self.tune_epoch = getattr(args, 'tune_epoch', 10000000)
        self.current_epoch = 0
        self.finetuning = False
        self.flip_switch = True

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.dictionary = dictionary
        self.pretrain_path = pretrain_path

        self.mask_factor = None
        self.embedding = self.build_embed(
            num_embeddings, self.pretrain_dim, padding_idx, dictionary, pretrain_path, self.tune_epoch)

    def hook(self, grad):
        grad *= self.mask_factor.type_as(grad)
        return grad

    def build_embed(self, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, tune_epoch):
        print(f'| Load embedding at {pretrain_path}')
        emb = transformer.Embedding(num_embeddings, embedding_dim, padding_idx)
        embed_dict = utils.parse_embedding(pretrain_path)

        self.mask_factor = []
        for idx in range(len(dictionary)):
            token = dictionary[idx]
            if token in embed_dict:
                emb.weight.data[idx] = embed_dict[token]
                self.mask_factor.append(0)
            else:
                self.mask_factor.append(1)

        self.mask_factor = torch.tensor(self.mask_factor, dtype=torch.float32).unsqueeze_(-1).expand(
            len(self.mask_factor), embedding_dim)

        emb.weight.register_hook(self.hook)
        return emb

    def turn_finetune(self, on=True):
        self.finetuning = on
        mess = "ON" if self.finetuning else "OFF"
        print(f'Turn {mess} finetuning')
        self.flip_switch = True

    def extra_repr(self):
        return f'pretrain_dim={self.pretrain_dim},tune_epoch={self.tune_epoch}'

    def _apply_embed(self, x, indices):
        bsz, node_len = x.size()
        assert bsz == 1
        leave_mask = (indices[:, :, 0] == indices[:, :, 1]) * x.ne(self.padding_idx)
        leaves = x[leave_mask].view(1, -1)

        _bsz, seq_len = leaves.size()
        rmask = torch.arange(0, seq_len, device=x.device).unsqueeze_(0).unsqueeze_(0).expand(bsz, node_len, seq_len)
        left_mask = rmask >= indices[:, :, :1]
        right_mask = rmask > indices[:, :, 1:]

        avg_mask = (left_mask ^ right_mask).unsqueeze_(-1)

        # avg_mask:     [b, nt, st, 1]
        # leaves:       [b, st]
        # o_leaves:     [b,  1, st]
        # leave_emb:    [b,  1, st, d]

        # embed_mesh:   [b, nt, st, d]
        # embed_out:    [b, nt, d]

        o_leaves = leaves.unsqueeze(1)
        leave_emb = self.embedding(o_leaves)

        avg_mask = avg_mask.type_as(leave_emb)
        avg_mask_sum = torch.max(avg_mask.sum(2), other=torch.tensor(1.0, device=avg_mask.device, dtype=avg_mask.dtype))
        embed_mesh = avg_mask * leave_emb
        embed_out = embed_mesh.sum(2) / avg_mask_sum

        return embed_out

    def _retrieve_leaves(self, x, indices):
        bsz, node_len = x.size()
        seq_len = (node_len + 1) // 2
        x_chunks = x.chunk(bsz, dim=0)
        idx_chunks = indices.chunk(bsz, dim=0)

        leaves = x.new(bsz, seq_len).fill_(self.padding_idx)

        def assign_sent_leave(i, seg, idx):
            mask = (idx[:, :, 0] == idx[:, :, 1]) * seg.ne(self.padding_idx)
            _lev = seg[mask]
            leaves[i][:_lev.size(0)].copy_(_lev)

        for _i, (_seg, _idx) in enumerate(zip(x_chunks, idx_chunks)):
            assign_sent_leave(_i, _seg, _idx)

        # leave_mask = (indices[:, :, 0] == indices[:, :, 1]) * x.ne(self.padding_idx)
        # leaves = x[leave_mask].view(1, -1)
        return leaves

    def forward(self, x, indices):
        """

        :param x:           [b, t]
        :param indices:     [b, t, 2]
        :return:
        """
        bsz, node_len = x.size()
        # x_chunks = x.chunk(bsz, dim=0)
        # idx_chunks = indices.chunk(bsz, dim=0)

        # embeds = [self._apply_embed(x, y) for x, y in zip(x_chunks, idx_chunks)]
        # embed_out = torch.cat(embeds, 0)

        # leave_mask = (indices[:, :, 0] == indices[:, :, 1]) * x.ne(self.padding_idx)
        # leaves = x[leave_mask].view()
        # # seq_len = (node_len - 1) // 2
        leaves = self._retrieve_leaves(x, indices)
        _bsz, seq_len = leaves.size()
        # rmask = torch.arange(0, seq_len).unsqueeze_(0).expand(node_len, seq_len)
        rmask = torch.arange(0, seq_len, device=x.device).unsqueeze_(0).unsqueeze_(0).expand(bsz, node_len, seq_len)
        left_mask = rmask >= indices[:, :, :1]
        right_mask = rmask > indices[:, :, 1:]
        avg_mask = (left_mask ^ right_mask).unsqueeze_(-1).type_as(leaves)

        # avg_mask:     [b, nt, st, 1]
        # leaves:       [b, st]
        # o_leaves:     [b,  1, st]
        # leave_emb:    [b,  1, st, d]

        # embed_mesh:   [b, nt, st, d]
        # embed_out:    [b, nt, d]

        o_leaves = leaves.unsqueeze(1)
        leave_emb = self.embedding(o_leaves)

        avg_mask = avg_mask.type_as(leave_emb)
        avg_mask_sum = torch.max(avg_mask.sum(2), other=torch.tensor(1.0, device=avg_mask.device, dtype=avg_mask.dtype))
        embed_mesh = avg_mask * leave_emb
        embed_out = embed_mesh.sum(2) / avg_mask_sum
        return embed_out


class PhraseAverageFinetunePretrainedEmbedding(PhraseAveragePretrainedEmbedding):
    def hook(self, grad):
        if self.flip_switch:
            print(f'Flip switch: {self.finetuning}')
            self.flip_switch = False
        if not self.finetuning:
            grad *= self.mask_factor.type_as(grad)
        return grad


class BertEmbedding(nn.Module):

    def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, freeze=True):
        super().__init__()

        self.args = args
        # self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
        self.tune_epoch = getattr(args, 'tune_epoch', 10000000)
        self.bert_name = getattr(args, 'bert_name', 'bert-base-uncased')
        self.bert_layer = getattr(args, 'bert_layer', 11)
        self.freeze = freeze

        self.current_epoch = 0
        self.finetuning = False
        self.flip_switch = True

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.dictionary = dictionary
        self.pretrain_path = pretrain_path

        self.unknown_idx = None
        self.mask_factor = None

        """
        dictionary: {word: idx}
        bert:       {word: idx}
        """

        self.pretrain_dim = self.get_pretrain_dim()
        self.index_remapping, self.bert_model = self.build_bert_dict_remapping()

        if self.embedding_dim != self.pretrain_dim:
            self.reproj = transformer.Linear(self.pretrain_dim, self.embedding_dim)
        else:
            self.reproj = lambda x: x

        self.embedding = Embedding(num_embeddings, self.embedding_dim, padding_idx)
        self.weight = self.embedding.weight

    def extra_repr(self):
        return 'bert_name={},bert_layer={},freeze={}'.format(
            self.bert_name, self.bert_layer, self.freeze
        )

    def build_bert_dict_remapping(self):
        """
        dictionary: {word: idx}
        bert:       {word: idx}
        """
        from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
        tokenizer = BertTokenizer.from_pretrained(self.bert_name, cache_dir=self.pretrain_path)
        model = BertModel.from_pretrained(self.bert_name, cache_dir=self.pretrain_path)

        for p in model.parameters():
            p.requires_grad = not self.freeze
        if self.freeze:
            model.eval()
        else:
            model.train()

        print(f'Finish loading model, retrieving vocab')
        words = deepcopy(self.dictionary.symbols)

        for i, v in enumerate(words):
            if v == '<pad>':
                words[i] = '[PAD]'
            if v == '<unk>':
                words[i] = '[UNK]'
                self.unknown_idx = i
            if v == '<Lua heritage>':
                words[i] = '[UNK]'

            if v == '</s>':
                words[i] = '[CLS]'

            if v not in tokenizer.vocab:
                words[i] = '[UNK]'

        assert self.unknown_idx is not None
        indexed_tokens = tokenizer.convert_tokens_to_ids(words)
        indexed_tokens_t = torch.tensor(indexed_tokens)
        remapping = nn.Parameter(indexed_tokens_t, requires_grad=False)

        unknown = indexed_tokens_t.eq(self.unknown_idx)
        print(f'Finish retriveing embeddings, unkown_idx={self.unknown_idx}, unkown_all ={unknown.sum()}')
        return remapping, model

    def get_pretrain_dim(self):
        if 'base' in self.bert_name:
            return 768
        else:
            raise NotImplementedError(f'name={self.bert_name}')

    def hook(self, grad):
        grad *= self.mask_factor.type_as(grad)
        return grad

    def forward(self, x, only_embedding=False):
        # x: [b, t]
        if only_embedding:
            return self.embedding(x)

        tensor = F.embedding(x, self.index_remapping)
        segments = torch.zeros_like(tensor)
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tensor, segments)
        embedding = encoded_layers[self.bert_layer]

        embedding = self.reproj(embedding)

        assert self.unknown_idx is not None
        unknown = x.eq(self.unknown_idx).unsqueeze_(-1)

        out_embed = torch.where(unknown, self.embedding(x), embedding)
        return out_embed


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    print(f'Embedding stats: max={m.weight.max()} - min={m.weight.min()} - mean={m.weight.mean()}')
    return m


def build_transformer_embedding(args, dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()

    if path:
        if args.pretrain_embed_mode == 'const':
            emb = PretrainedEmbedding(num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'sep':
            emb = SeparatePretrainedEmbedding(args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'sep_sdist':
            emb = SeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, same_dist=True)
        elif args.pretrain_embed_mode == 'sep_eunk':
            emb = SeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, enforce_unk=True)
        elif args.pretrain_embed_mode == 'sep_eunk_sdist':
            emb = SeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, enforce_unk=True, same_dist=True)
        elif args.pretrain_embed_mode == 'downscale_sep':
            emb = DownscaleSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'downscale_sep_sdist':
            emb = DownscaleSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, same_dist=True)

        elif args.pretrain_embed_mode == 'finetune_sep':
            emb = FinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, freeze=False, enforce_unk=False)
        elif args.pretrain_embed_mode == 'finetune_sep_eunk':
            emb = FinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path, freeze=False, enforce_unk=True)

        elif args.pretrain_embed_mode == 'downscale_finetune_sep':
            emb = DownscaleFinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path,
                freeze=False, enforce_unk=False, downscale_new=False)
        elif args.pretrain_embed_mode == 'downscale_finetune_sep_downnew':
            emb = DownscaleFinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path,
                freeze=False, enforce_unk=False, downscale_new=True)

        elif args.pretrain_embed_mode == 'downscale_finetune_sep_eunk':
            emb = DownscaleFinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path,
                freeze=False, enforce_unk=True, downscale_new=False)
        elif args.pretrain_embed_mode == 'downscale_finetune_sep_eunk_downnew':
            emb = DownscaleFinetuneSeparatePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path,
                freeze=False, enforce_unk=True, downscale_new=True)
        elif args.pretrain_embed_mode == 'linear_const':
            emb = LinearPretrainedEmbedding(args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'finetune':
            emb = FinetunePretrainedEmbedding(args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'free':
            emb = FreePretrainedEmbedding(num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'plb_avg_const':
            emb = PhraseAveragePretrainedEmbedding(args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'plb_avg_finetune':
            emb = PhraseAverageFinetunePretrainedEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        elif args.pretrain_embed_mode == 'bert':
            emb = BertEmbedding(
                args, num_embeddings, embed_dim, padding_idx, dictionary, path)
        else:
            raise NotImplementedError(f'args.pretrain_embed_mode = {args.pretrain_embed_mode} not iompl')
    else:
        print(f'Build new random Embeddings...')
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # print(f'Embedding stats: max={emb.weight.max()} - min={emb.weight.min()} - mean={emb.weight.mean()}')
    return emb
