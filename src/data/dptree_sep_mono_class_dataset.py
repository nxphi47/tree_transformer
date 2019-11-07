import numpy as np
import torch
from fairseq import utils

from . import data_utils, FairseqDataset

from fairseq.data import monolingual_dataset, language_pair_dataset

from . import DPTreeMonoClassificationDataset


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    nsent = max(v.size(0) for v in values)
    size = max(v.size(1) for v in values)
    res = values[0].new(len(values), nsent, size).fill_(pad_idx)

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


def dptree_sep_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=False,
):
    assert not left_pad_source
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

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
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
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


class DPTreeSeparateMonoClassificationDataset(DPTreeMonoClassificationDataset):

    def __init__(self, srcs, src_sizes, src_dict, tgt=None, left_pad_source=False, max_source_positions=1024,
                 max_target_positions=1024, shuffle=True, input_feeding=False, remove_eos_from_source=False):
        super().__init__(srcs, src_sizes, src_dict, tgt, left_pad_source, max_source_positions, max_target_positions,
                         shuffle, input_feeding, remove_eos_from_source)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index]

    def collater(self, samples):
        return dptree_sep_collate(
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
                'source': self._get_dummy_source_example(src_len),
                'target': torch.zeros(1, dtype=torch.long),
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
            'nodes': nodes.unsqueeze_(0),
            'labels': labels.unsqueeze_(0),
            'indices': indices.unsqueeze_(0),
            'length': length
        }
        return example


def dptree_sep_collate_node(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=False,
):
    assert not left_pad_source
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source'].keys()}

        nodes = src['nodes']
        labels = src['labels']
        indices = src['indices']
        length = src['length']

        nodes = collate_token_list(nodes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        labels = collate_token_list(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        indices = dptree_collate_sep_indices(indices, 0, 0, left_pad, move_eos_to_beginning)
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
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
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
        'target': src['labels'],
    }
    # sizes = {k: v.size() for k, v in batch['net_input'].items()}
    # print(f'batch-net-inputs: {sizes}')

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    # print(f'Collate after: {batch["net_input"]["src_labels"]}')
    return batch


class DPTreeSeparateNodeMonoClassificationDataset(DPTreeSeparateMonoClassificationDataset):
    """
    Training on subtree labels, test on root tree
    """

    PLACEHOLDER_ID = 4

    def collater(self, samples):
        return dptree_sep_collate_node(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def _get_dummy_source_example(self, src_len):
        # 'nodes', 'labels', 'indices', 'length']
        nodes = self.src_dict.dummy_sentence(src_len)
        labels = torch.ones_like(nodes)
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

    @property
    def supports_prefetch(self):
        return getattr(self.src, 'supports_prefetch', False)

    def prefetch(self, indices):
        # self.src.prefetch(indices)
        for k, v in self.srcs.items():
            v.prefetch(indices)
        # self.tgt.prefetch(indices)


def collate_dptree_sep_nli(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=False,
):
    assert not left_pad_source
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        _src = {k: [dic['source'][0][k] for dic in samples] for k in samples[0]['source'][0]}
        _src2 = {k: [dic['source'][1][k] for dic in samples] for k in samples[0]['source'][0]}

        def acquire_object(x):
            nodes = x['nodes']
            labels = x['labels']
            indices = x['indices']
            length = x['length']

            nodes = collate_token_list(nodes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
            labels = collate_token_list(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
            indices = dptree_collate_sep_indices(indices, 0, 0, left_pad, move_eos_to_beginning)
            length = data_utils.collate_tokens(length, 0, 0, False)

            src_o = {
                'nodes': nodes,
                'labels': labels,
                'indices': indices,
                'length': length
            }
            return src_o

        out_srcs = (acquire_object(_src), acquire_object(_src2))
        return out_srcs

    id = torch.LongTensor([s['id'] for s in samples])
    src, src2 = merge_source(left_pad_source)
    src_lengths = torch.LongTensor([s['source'][0]['nodes'].numel() for s in samples])
    src_lengths2 = torch.LongTensor([s['source'][1]['nodes'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    # reorder
    src = {k: v.index_select(0, sort_order) for k, v in src.items()}
    src2 = {k: v.index_select(0, sort_order) for k, v in src2.items()}

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
    else:
        # ntokens = sum(len(s['source']) for s in samples)
        ntokens = sum(len(s['source'][0]['nodes']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            # 'src_tokens': (src['nodes'], src2['nodes']),
            # 'src_labels': (src['labels'], src2['labels']),
            # 'src_indices': (src['indices'], src2['indices']),
            # 'src_sent_lengths': (src['length'], src2['length']),
            # 'src_lengths': (src_lengths, src_lengths2),
            'src_tokens1': src['nodes'],
            'src_tokens2': src2['nodes'],
            'src_labels1': src['labels'],
            'src_labels2': src2['labels'],
            'src_indices1': src['indices'],
            'src_indices2': src2['indices'],
            'src_sent_lengths1': src['length'],
            'src_sent_lengths2': src2['length'],
            'src_lengths1': src_lengths,
            'src_lengths2': src_lengths2,

        },
        'target': target,
    }
    # sizes = {k: v.size() for k, v in batch['net_input'].items()}
    # print(f'batch-net-inputs: {sizes}')
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class DPTreeSeparateLIClassificationDataset(FairseqDataset):

    def __init__(
            self,
            srcs, src_sizes, srcs2, src_sizes2,
            src_dict,
            tgt=None,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):
        self.srcs = srcs
        self.srcs2 = srcs2
        self.src = srcs['nodes']
        self.src2 = srcs2['nodes']

        self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        self.src_sizes2 = np.array(src_sizes2)
        assert len(self.src_sizes) == len(self.src), f'{len(self.src_sizes)} = {len(self.src)}'

        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = {k: v[index] for k, v in self.srcs.items()}
        src_item2 = {k: v[index] for k, v in self.srcs2.items()}

        if self.remove_eos_from_source:
            raise NotImplementedError(
                f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': (src_item, src_item2),
            'target': tgt_item,
        }

    def __len__(self):
        assert len(self.src) == len(self.tgt), f'{len(self.src)} != {len(self.tgt)}'
        assert len(self.src) == len(self.src2), f'{len(self.src)} != {len(self.src2)}'
        return len(self.src)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        return self.src_sizes[index] + self.src_sizes2[index]

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        # return (self.src_sizes[index], 0)
        return self.src_sizes[index] + self.src_sizes2[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        return indices[np.argsort(self.src_sizes[indices] + self.src_sizes2[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and getattr(self.src2, 'supports_prefetch', False)
                and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        for src in [self.srcs, self.srcs2]:
            for k, v in src.items():
                v.prefetch(indices)
        self.tgt.prefetch(indices)

    def collater(self, samples):
        return collate_dptree_sep_nli(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            # left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        bsz = max(num_tokens // src_len, 1)
        return self.collater([
            {
                'id': i,
                'source': (self._get_dummy_source_example(src_len), self._get_dummy_source_example(src_len)),
                'target': torch.zeros(1, dtype=torch.long),
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
            'nodes': nodes.unsqueeze_(0),
            'labels': labels.unsqueeze_(0),
            'indices': indices.unsqueeze_(0),
            'length': length
        }
        return example





