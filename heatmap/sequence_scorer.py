# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import os
import torch.nn.functional as F
from fairseq import utils

from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
from ..src.nstack2seq_generator import Nstack2SeqGenerator

GET_ENCOUT = bool(int(os.environ.get('get_encout', 0)))
GET_INNER_ATT = bool(int(os.environ.get('get_inner_att', 0)))
INNER_ATT = int(os.environ.get('inner_att', -1))


class HeatmapSequenceScorer(object):
    """Scores the target for a given source sentence."""

    def __init__(self, models, pad):
        self.models = models
        # self.pad = tgt_dict.pad()
        self.pad = pad

    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def score_batched_itr(self, data_itr, cuda=False, timer=None):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if timer is not None:
                timer.start()
            pos_scores, attn = self.score(s)
            for i, id in enumerate(s['id'].data):
                # remove padding from ref
                src = utils.strip_pad(s['net_input']['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                tgt_len = ref.numel()
                pos_scores_i = pos_scores[i][:tgt_len]
                score_i = pos_scores_i.sum() / tgt_len
                if attn is not None:
                    attn_i = attn[i]
                    _, alignment = attn_i.max(dim=0)
                else:
                    attn_i = alignment = None
                hypos = [{
                    'tokens': ref,
                    'score': score_i,
                    'attention': attn_i,
                    'alignment': alignment,
                    'positional_scores': pos_scores_i,
                }]
                if timer is not None:
                    timer.stop(s['ntokens'])
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos

    def score(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        for model in self.models:
            with torch.no_grad():
                model.eval()
                # decoder_out = model.forward(**net_input)
                prev_output_tokens = net_input['prev_output_tokens']
                del net_input['prev_output_tokens']
                encoder_out = model.encoder(**net_input)
                decoder_out = model.decoder(prev_output_tokens, encoder_out)
                # return decoder_out
                if GET_ENCOUT:
                    attn = F.softmax(100 * encoder_out['encoder_out'].transpose(0, 1), 1).mean(-1)
                    bsz, tk = attn.size()
                    tq = prev_output_tokens.size(1)
                    attn = attn.unsqueeze_(1).expand(bsz, tq, tk)
                    cross_attn = decoder_out[1]['attn']
                    assert list(attn.size()) == list(cross_attn.size()), f'{attn.size()} != {cross_attn.size()}, {prev_output_tokens.size()}'
                    # attn:     [b, tk, C]
                else:
                    attn = decoder_out[1]

            probs = model.get_normalized_probs(decoder_out, log_probs=len(self.models) == 1, sample=sample).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)

            if attn is not None:
                # {'attn': attn, 'inner_states': inner_states}
                if not torch.is_tensor(attn):
                    if GET_INNER_ATT:
                        attn = attn['inner_atts'][INNER_ATT]
                    else:
                        attn = attn['attn']

                assert torch.is_tensor(attn), f'attn: {attn}'
                attn = attn.data
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        if len(self.models) > 1:
            avg_probs.div_(len(self.models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(self.models))
        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )
        return avg_probs.squeeze(2), avg_attn


class HeatmapSequenceAttentionEntropyScorer(HeatmapSequenceScorer):
    def score(self, sample):
        """Score a batch of translations."""
        net_input = sample['net_input']

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_attn = None
        assert len(self.models) == 1, f'{len(self.models)} not 1'
        for model in self.models:
            with torch.no_grad():
                model.eval()
                # decoder_out = model.forward(**net_input)
                prev_output_tokens = net_input['prev_output_tokens']
                del net_input['prev_output_tokens']
                encoder_out = model.encoder(**net_input)
                decoder_out = model.decoder(prev_output_tokens, encoder_out)
                # return decoder_out
                if GET_ENCOUT:
                    # attn = F.softmax(100 * encoder_out['encoder_out'].transpose(0, 1), 1).mean(-1)
                    # bsz, tk = attn.size()
                    # tq = prev_output_tokens.size(1)
                    # attn = attn.unsqueeze_(1).expand(bsz, tq, tk)
                    # cross_attn = decoder_out[1]['attn']
                    # assert list(attn.size()) == list(cross_attn.size()), f'{attn.size()} != {cross_attn.size()}, {prev_output_tokens.size()}'
                    # # attn:     [b, tk, C]
                    raise NotImplementedError
                else:
                    attn = decoder_out[1]

            probs = model.get_normalized_probs(decoder_out, log_probs=len(self.models) == 1, sample=sample).data
            if avg_probs is None:
                avg_probs = probs
            else:
                avg_probs.add_(probs)

            assert 'inner_atts' in attn
            inner_atts = attn['inner_atts']
            avg_attn = inner_atts[-1]
            # [b, tq, tk]
            inner_att_entropies = [-(x * x.log()).sum(dim=-1) for x in inner_atts]
            # [b, tq]

            inner_atts = torch.cat([x.unsqueeze_(-1) for x in inner_atts], dim=-1)
            inner_att_entropies = torch.cat([x.unsqueeze_(-1) for x in inner_att_entropies], dim=-1)
            # [b, tq, tk, 6]
            # [b, tq, 6]

            # if attn is not None:
            #     # {'attn': attn, 'inner_states': inner_states}
            #     if not torch.is_tensor(attn):
            #         if GET_INNER_ATT:
            #             attn = attn['inner_atts'][INNER_ATT]
            #         else:
            #             attn = attn['attn']
            #
            #     assert torch.is_tensor(attn), f'attn: {attn}'
            #     attn = attn.data
            #     if avg_attn is None:
            #         avg_attn = attn
            #     else:
            #         avg_attn.add_(attn)

        if len(self.models) > 1:
            avg_probs.div_(len(self.models))
            avg_probs.log_()
            if avg_attn is not None:
                avg_attn.div_(len(self.models))

        avg_probs = avg_probs.gather(
            dim=2,
            index=sample['target'].data.unsqueeze(-1),
        )
        return avg_probs.squeeze(2), avg_attn, inner_atts, inner_att_entropies

    def score_batched_itr(self, data_itr, cuda=False, timer=None):
        """Iterate over a batched dataset and yield scored translations."""
        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if timer is not None:
                timer.start()
            pos_scores, attn, inner_atts, inner_att_entropies = self.score(s)
            for i, id in enumerate(s['id'].data):
                # remove padding from ref
                src = utils.strip_pad(s['net_input']['src_tokens'].data[i, :], self.pad)
                ref = utils.strip_pad(s['target'].data[i, :], self.pad) if s['target'] is not None else None
                tgt_len = ref.numel()
                pos_scores_i = pos_scores[i][:tgt_len]
                score_i = pos_scores_i.sum() / tgt_len
                if attn is not None:
                    attn_i = attn[i]
                    _, alignment = attn_i.max(dim=0)
                else:
                    attn_i = alignment = None

                inner_att = inner_atts[i]
                inner_att_entropy = inner_att_entropies[i]

                hypos = [{
                    'tokens': ref,
                    'score': score_i,
                    'attention': attn_i,
                    'inner_att': inner_att,
                    'inner_att_entropy': inner_att_entropy,
                    'alignment': alignment,
                    'positional_scores': pos_scores_i,
                }]
                if timer is not None:
                    timer.stop(s['ntokens'])
                # return results in the same format as SequenceGenerator
                yield id, src, ref, hypos


class Nstack2SeqHeatmapScorer(object):

    def __init__(self, generator, image_dir, **kwargs) -> None:
        super().__init__()
        self.generator = generator
        self.image_dir = image_dir

    def generate(
            self,
            models,
            sample,
            prefix_tokens=None,
            bos_token=None,
            **kwargs
    ):
      hypos = self.generator.generate(models, sample, prefix_tokens=prefix_tokens)
      target = sample['target']

      flipped_src_tokens = sample['net_input']['src_tokens']
      src_tokens = torch.flip(flipped_src_tokens, [2])
      attention = hypos['attention']

      # src_tokens = hypos['tokens']
      """
      'tokens': tokens_clone[i],
        'score': score,
        'attention': hypo_attn,  # src_len x tgt_len
        'alignment': alignment,
        'positional_scores': pos_scores[i],
      """

      assert src_tokens.size(0) == 1, f'bsz should be 1 ->{src_tokens.size()}'



