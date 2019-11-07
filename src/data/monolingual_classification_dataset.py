import numpy as np
import torch
from fairseq import utils

from . import data_utils, FairseqDataset
from fairseq.data import Dictionary
from fairseq.data import monolingual_dataset, language_pair_dataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        values = [s[key] for s in samples]
        assert values[0][-1] == eos_idx, f'eos_idx={eos_idx},values={values[0]}'
        return data_utils.collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    assert src_tokens[0][-1] == eos_idx, f'eos_idx={eos_idx}, src_tokens[0] = {src_tokens[0]}'
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
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
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class MonolingualClassificationDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

        # tgt_sizes (List[int], optional): target sentence lengths
        # tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).

        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None,
        left_pad_source=True,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):
        # if tgt_dict is not None:
        #     assert src_dict.pad() == tgt_dict.pad()
        #     assert src_dict.eos() == tgt_dict.eos()
        #     assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        # self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        # self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        # self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        # self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        eos = self.src_dict.eos()
        assert self.src[index][-1] == eos

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

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
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            # left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        bsz = max(num_tokens // src_len, 1)
        bsz = 1
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                # 'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'target': torch.zeros(1, dtype=torch.long),
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], 0)
        return self.src_sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # if self.tgt_sizes is not None:
        #     indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)


def collate_nli(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        values = [s[key] for s in samples]
        assert values[0][-1] == eos_idx, f'eos_idx={eos_idx},values={values[0]}'
        return data_utils.collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

    def merge_source(left_pad, move_eos_to_beginning=False):
        key = 'source'
        _src1 = [s[key][0] for s in samples]
        _src2 = [s[key][1] for s in samples]

        _src1 = data_utils.collate_tokens(_src1, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        _src2 = data_utils.collate_tokens(_src2, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        return (_src1, _src2)

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    id = torch.LongTensor([s['id'] for s in samples])
    # src_tokens = merge('source', left_pad=left_pad_source)
    src1, src2 = merge_source(left_pad=left_pad_source)
    # assert src_tokens[0][-1] == eos_idx, f'eos_idx={eos_idx}, src_tokens[0] = {src_tokens[0]}'
    # sort by descending source length
    # src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths1 = torch.LongTensor([s['source'][0].numel() for s in samples])
    src_lengths2 = torch.LongTensor([s['source'][1].numel() for s in samples])
    src_lengths = src_lengths1 + src_lengths2

    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src1 = src1.index_select(0, sort_order)
    src2 = src2.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
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
        ntokens = sum(len(s['source'][0]) + len(s['source'][1]) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            # 'src_tokens': src_tokens,
            'src_tokens1': src1,
            'src_tokens2': src2,
            'src_lengths1': src_lengths1,
            'src_lengths2': src_lengths2,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def collate_nli_concat(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        values = [s[key] for s in samples]
        assert values[0][-1] == eos_idx, f'eos_idx={eos_idx},values={values[0]}'
        return data_utils.collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

    def merge_source(left_pad, move_eos_to_beginning=False):
        key = 'source'
        _src1 = [s[key][0] for s in samples]
        _src2 = [s[key][1] for s in samples]
        _src = [torch.cat([x, y], 0) for x, y in zip(_src1, _src2)]
        # assert left_pad
        assert all(x[-1] == eos_idx for x in _src), f'eos_idx={eos_idx},values={_src}'
        _src = data_utils.collate_tokens(_src, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

        # _src1 = data_utils.collate_tokens(_src1, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # _src2 = data_utils.collate_tokens(_src2, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        return _src

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    id = torch.LongTensor([s['id'] for s in samples])
    # src_tokens = merge('source', left_pad=left_pad_source)
    src1 = merge_source(left_pad=left_pad_source)
    # assert src_tokens[0][-1] == eos_idx, f'eos_idx={eos_idx}, src_tokens[0] = {src_tokens[0]}'
    # sort by descending source length
    # src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths1 = torch.LongTensor([s['source'][0].numel() for s in samples])
    src_lengths2 = torch.LongTensor([s['source'][1].numel() for s in samples])
    src_lengths = src_lengths1 + src_lengths2

    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src1 = src1.index_select(0, sort_order)
    # src2 = src2.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
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
        ntokens = sum(len(s['source'][0]) + len(s['source'][1]) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            # 'src_tokens': src_tokens,
            'src_tokens1': src1,
            # 'src_tokens2': src2,
            'src_lengths1': src_lengths,
            # 'src_lengths2': src_lengths2,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class A():
    def o(self, x, y, **kwargs):
        return x + y
    def o(self, z, **kwargs):
        return z * 10

# a = A()
# a.o(**{'x': 12, 'y': 9})
# a.o(**{'z': 3})


class MonolingualNLIClassificationDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

        # tgt_sizes (List[int], optional): target sentence lengths
        # tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing
            (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).

        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """
    def __init__(
        self, src1, src2, src_sizes1, src_sizes2, src_dict,
        tgt=None,
        left_pad_source=True,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):
        # if tgt_dict is not None:
        #     assert src_dict.pad() == tgt_dict.pad()
        #     assert src_dict.eos() == tgt_dict.eos()
        #     assert src_dict.unk() == tgt_dict.unk()
        self.src = src1
        self.src1 = src1
        self.src2 = src2
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes1)
        self.src_sizes1 = np.array(src_sizes1)
        self.src_sizes2 = np.array(src_sizes2)

        # self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        # self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        # self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        # self.append_eos_to_target = append_eos_to_target

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item1 = self.src1[index]
        src_item2 = self.src2[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            # eos = self.src_dict.eos()
            # if self.src[index][-1] == eos:
            #     src_item = self.src[index][:-1]
            raise NotImplementedError

        return {
            'id': index,
            'source': (src_item1, src_item2),
            'target': tgt_item,
        }

    def __len__(self):
        assert len(self.src1) == len(self.tgt), f'{len(self.src1)} != {len(self.tgt)}'
        assert len(self.src1) == len(self.src2), f'{len(self.src1)} != {len(self.src2)}'
        return len(self.src1)

    def collater(self, samples):
        return collate_nli(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        bsz = max(num_tokens // src_len, 1)
        return self.collater([
            {
                'id': i,
                'source': (self.src_dict.dummy_sentence(src_len), self.src_dict.dummy_sentence(src_len)),
                # 'source': (self._get_dummy_source_example(src_len), self._get_dummy_source_example(src_len)),
                'target': torch.zeros(1, dtype=torch.long),
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes1[index] + self.src_sizes2[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], 0)
        return self.src_sizes1[index] + self.src_sizes2[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        # if self.tgt_sizes is not None:
        #     indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices] + self.src_sizes2[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src1, 'supports_prefetch', False)
            and getattr(self.src2, 'supports_prefetch', False)
            and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        self.src1.prefetch(indices)
        self.src2.prefetch(indices)
        self.tgt.prefetch(indices)


class MonolingualNLIConcatClassificationDataset(MonolingualNLIClassificationDataset):
    def collater(self, samples):
        return collate_nli_concat(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )


class BertDictionary(Dictionary):

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', class_word='<CLASS>'):
        super().__init__(pad, eos, unk)
        self.class_word = class_word
        self.class_index = self.add_symbol(self.class_word)
        self.nspecial = len(self.symbols)

    def cls(self):
        """Helper to get index of pad symbol"""
        return self.class_index


class MonolingualNLIConcatBertClassificationDataset(MonolingualNLIClassificationDataset):
    CLASS_SYMBOL = '<CLASS>'

    def collater(self, samples):
        return collate_nli_concat(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item1 = self.src1[index]
        src_item2 = self.src2[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # # if self.append_eos_to_target:
        # eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        # #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        cls = self.src_dict.cls()
        src_item1 = torch.cat([torch.LongTensor([cls]), src_item1])

        if self.remove_eos_from_source:
            # eos = self.src_dict.eos()
            # if self.src[index][-1] == eos:
            #     src_item = self.src[index][:-1]
            raise NotImplementedError

        return {
            'id': index,
            'source': (src_item1, src_item2),
            'target': tgt_item,
        }



