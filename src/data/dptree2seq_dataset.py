import numpy as np
import torch

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset

DPTREE_KEYS = ['nodes', 'labels', 'indices', 'length']


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def dptree2seq_collate_indices(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning):
    """convert list of 2d tensors into padded 3d tensors"""
    assert not left_pad
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size, 2).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])

    return res


def dptree2seq_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True,
):
    if len(samples) == 0:
        return {}

    # print(samples)
    # raise NotImplementedError

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_source(left_pad, move_eos_to_beginning=False):
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
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
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


class DPTree2SeqPairDataset(FairseqDataset):
    def __init__(
            self, srcs, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False,
            append_eos_to_target=False,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.srcs = srcs
        self.src = srcs['nodes']
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        same = self.tgt_dict.eos() == self.src_dict.eos()
        print(f'| ATTENTION ! EOS same: {same}')

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        # src_item = self.src[index]
        src_item = {k: v[index] for k, v in self.srcs.items()}

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            # if self.tgt_dict:
            #     # same = self.tgt_dict.eos() == self.src_dict.eos()
            #     # print(f'EOS same: {same}')
            #     eos = self.tgt_dict.eos()
            # else:
            #     eos = self.src_dict.eos()

            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:

            # eos = self.src_dict.eos()
            # if self.src[index][-1] == eos:
            #     src_item = self.src[index][:-1]
            raise NotImplementedError(f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

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
        return dptree2seq_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self._get_dummy_source_example(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

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
            'nodes': nodes,
            'labels': labels,
            'indices': indices,
            'length': length
        }
        return example

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def prefetch(self, indices):
        # self.src.prefetch(indices)
        print(f'| {self.__class__.__name__}:prefetch:starting...')
        for k, v in self.srcs.items():
            v.prefetch(indices)
            print(f'| {self.__class__.__name__}:prefetch:{k}')
        print(f'| {self.__class__.__name__}:prefetch:tgt')
        self.tgt.prefetch(indices)
        print(f'| {self.__class__.__name__}:prefetch:finished...')

    @property
    def supports_prefetch(self):
        return (
            hasattr(self.src, 'supports_prefetch')
            and self.src.supports_prefetch
            and hasattr(self.tgt, 'supports_prefetch')
            and self.tgt.supports_prefetch
        )


