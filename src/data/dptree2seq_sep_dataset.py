import numpy as np
import torch

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset
from .dptree2seq_dataset import *


def collate_token_list(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    try:
        nsent = max(v.size(0) for v in values)
        size = max(v.size(1) for v in values)
        res = values[0].new(len(values), nsent, size).fill_(pad_idx)
    except RuntimeError as e:
        print(f'values: {[v.size() for v in values]}')
        raise e

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f'{res.size()}, {src.size()}, {dst.size()}'
        if move_eos_to_beginning:
            # assert src[-1] == eos_idx
            # dst[0] = eos_idx
            # dst[1:] = src[:-1]
            raise NotImplementedError
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        # dest = res[i][size - len(v):] if left_pad else res[i][:len(v)]
        nsent, length = v.size()
        if left_pad:
            dest = res[i, :nsent, size - len(v[0]):]
        else:
            dest = res[i, :nsent, :len(v[0])]
        copy_tensor(v, dest)
    return res


def dptree_collate_sep_indices(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
    """convert list of 2d tensors into padded 3d tensors"""
    assert not left_pad
    nsent = max(v.size(0) for v in values)
    size = max(v.size(1) for v in values)
    res = values[0].new(len(values), nsent, size, 2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f'{res.size()}, {src.size()}, {dst.size()}'
        if move_eos_to_beginning:
            # assert src[-1] == eos_idx
            # dst[0] = eos_idx
            # dst[1:] = src[:-1]
            raise NotImplementedError
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        nsent, length, _ = v.size()
        copy_tensor(v, res[i, :nsent, size - len(v[0]):] if left_pad else res[i, :nsent, :len(v[0])])

    return res


def dptree2seq_sep_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True,
):
    if len(samples) == 0:
        return {}

    # print(samples)
    # raise NotImplementedError

    def merge_target(key, left_pad, move_eos_to_beginning=False):
        try:
            output = data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx, eos_idx, left_pad, move_eos_to_beginning,
            )
        except AssertionError as ae:
            print([s[key][-1] for s in samples])
            print([len(s[key]) for s in samples])
            raise ae
        return output

    def merge_source_backup(left_pad, move_eos_to_beginning=False):
        # src = [s['source'] for s in samples]
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        nodes = src['nodes']
        labels = src['labels']
        indices = src['indices']
        length = src['length']

        nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        labels = data_utils.collate_tokens(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        indices = dptree2seq_collate_indices(indices, 0, 0, left_pad, move_eos_to_beginning)
        length = torch.cat([x.unsqueeze_(0) for x in length], 0)

        src_o = {
            'nodes': nodes,
            'labels': labels,
            'indices': indices,
            'length': length
        }
        return src_o

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        nodes = src['nodes']
        labels = src['labels']
        indices = src['indices']
        length = src['length']

        nodes = collate_token_list(nodes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        labels = collate_token_list(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        indices = dptree_collate_sep_indices(indices, 0, 0, left_pad, move_eos_to_beginning)
        # length = torch.cat([x.unsqueeze_(0) for x in length], 0)
        length = data_utils.collate_tokens(length, 0, 0, False)

        src_o = {
            'nodes': nodes,
            'labels': labels,
            'indices': indices,
            'length': length
        }
        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source(left_pad_source)
    src_lengths = torch.LongTensor([s['source']['nodes'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    # reoreder
    src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_target('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge_target(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']['nodes']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src['nodes'],
            'src_labels': src['labels'],
            'src_indices': src['indices'],
            'src_sent_lengths': src['length'],
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    # sizes = {k: v.size() for k, v in batch['net_input'].items()}
    # print(f'batch-net-inputs: {sizes}')
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class DPTree2SeqSeparatePairDataset(DPTree2SeqPairDataset):
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return dptree2seq_sep_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def _get_dummy_source_example(self, src_len):
        # 'nodes', 'labels', 'indices', 'length']
        nodes = self.src_dict.dummy_sentence(src_len)
        labels = self.src_dict.dummy_sentence(src_len)
        node_len = nodes.size()[0]
        seq_len = int((node_len + 1) // 2)  # w/o pad

        # t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        length = torch.tensor([seq_len]).long()
        # test dummy zeros for indices
        fl_indices = torch.arange(node_len).long().unsqueeze(1)
        row_indices = fl_indices // seq_len
        col_indices = fl_indices - row_indices * seq_len
        indices = torch.cat([row_indices, col_indices], 1)

        example = {
            'nodes': nodes.unsqueeze_(0),
            'labels': labels.unsqueeze_(0),
            'indices': indices.unsqueeze_(0),
            'length': length
        }
        return example
