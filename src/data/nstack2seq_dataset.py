import numpy as np
import torch

from fairseq import utils

from fairseq.data import data_utils, FairseqDataset
from .dptree2seq_dataset import *
from .dptree2seq_sep_dataset import *
from .nstack_mono_class_dataset import *


def nstack2seq_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True, is_infer=False):
    assert not left_pad_source
    if len(samples) == 0:
        return {}
    if len(samples) == 0:
        return {}

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

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        s_leaves = collate_nstack_leaves(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        s_pos_tags = collate_nstack_leaves(pos_tags, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        s_nodes, s_spans = collate_nstack_rv_nodes(nodes, spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

        b, n, _ = s_leaves.size()
        lengths = torch.zeros(b, n, device=s_leaves.device).int()

        src_o = {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,

            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,

            'node_indices': s_spans,

            'length': lengths
        }

        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source(left_pad_source)
    src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    if is_infer:
        src_tokens = torch.cat([torch.flip(src['node_nodes'], [2]), src['node_leaves']], 2)
        src_labels = torch.cat([torch.flip(src['label_nodes'], [2]), src['label_leaves']], 2)
    else:
        # fixme: to ensure integrity with previous codes
        src_tokens = torch.flip(torch.cat([src['node_leaves'], src['node_nodes']], 2), [2])
    # src_tokens = torch.flip(src_tokens, [2])
        src_labels = torch.flip(torch.cat([src['label_leaves'], src['label_nodes']], 2), [2])
    # src_labels = torch.flip(src_labels, [2])

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
            'src_node_leaves': src['node_leaves'],
            'src_node_nodes': src['node_nodes'],

            'src_label_leaves': src['label_leaves'],
            'src_label_nodes': src['label_nodes'],

            'src_node_indices': src['node_indices'],

            'src_sent_lengths': src['length'],
            'src_lengths': src_lengths,

            'src_tokens': src_tokens,
            'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class Nstack2SeqPairDataset(FairseqDataset):
    def __init__(
            self, srcs, src_sizes, src_dict, src_nsents,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False,
            append_eos_to_target=False, is_infer=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.srcs = srcs
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.span_data = srcs['spans']

        self.tgt = tgt
        self.src_nsents = np.array(src_nsents)
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
        self.is_infer = is_infer
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
        return nstack2seq_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            is_infer=self.is_infer
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
        # ['leaves', 'nodes', 'pos_tags', 'spans']
        leave_len = (src_len + 1) // 2
        node_len = src_len - leave_len

        leaves = self.src_dict.dummy_sentence(leave_len)
        pos_tags = self.src_dict.dummy_sentence(leave_len)
        nodes = self.src_dict.dummy_sentence(node_len)
        spans = torch.tensor([0, leave_len - 1]).view(1, 1, 2).expand(1, node_len, 2)

        example = {
            'leaves': leaves.unsqueeze_(0),
            'pos_tags': pos_tags.unsqueeze_(0),
            'nodes': nodes.unsqueeze_(0),
            'spans': spans,
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

    def nsent(self, index):
        return self.src_nsents[index]

    def src_size_nsent(self, index):
        return (self.src_sizes[index], self.src_nsents[index])

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
            # print(f'| {self.__class__.__name__}:prefetch:{k}')
        # print(f'| {self.__class__.__name__}:prefetch:tgt')
        self.tgt.prefetch(indices)
        # print(f'| {self.__class__.__name__}:prefetch:finished...')

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.src, 'supports_prefetch')
                and self.src.supports_prefetch
                and hasattr(self.tgt, 'supports_prefetch')
                and self.tgt.supports_prefetch
        )


def collate_nstack_rv_nodes(values, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    assert not move_eos_to_beginning
    res, size, nsent = get_sep_res(values, pad_idx)
    res_idx = indices[0].new(len(values), nsent, size, 2).fill_(0)

    for i, (v, idx) in enumerate(zip(values, indices)):
        nsent = v.size(0)
        rv_nodes = torch.flip(v, [1])
        rv_idx = torch.flip(idx, [1])

        assert rv_nodes.size(1) == rv_idx.size(1), f'{rv_nodes.size()} != {rv_idx.size()}'
        dest = res[i, :nsent, size - rv_nodes.size(1):]
        idx_dest = res_idx[i, :nsent, size - rv_nodes.size(1):]
        copy_tensor(rv_nodes, dest, eos_idx, move_eos_to_beginning)
        copy_tensor(rv_idx, idx_dest, eos_idx, move_eos_to_beginning)

    return res, res_idx


def default_collate_spans(spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    size = max(x.size(0) for x in spans)
    res_idx = spans[0].new(len(spans), size, 2).fill_(0)

    for i, v in enumerate(spans):
        # copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        idx_dest = res_idx[i, size - len(v):]
        copy_tensor(v, idx_dest, eos_idx, move_eos_to_beginning)
    return res_idx


def nstackmerge2seq_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True, is_infer=False):
    assert not left_pad_source
    if len(samples) == 0:
        return {}
    if len(samples) == 0:
        return {}

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

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        s_leaves = data_utils.collate_tokens(leaves, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_pos_tags = data_utils.collate_tokens(pos_tags, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, True, move_eos_to_beginning)
        s_spans = default_collate_spans(spans, pad_idx, eos_idx, True, move_eos_to_beginning)

        # s_leaves = collate_nstack_leaves(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # s_pos_tags = collate_nstack_leaves(pos_tags, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # s_nodes, s_spans = collate_nstack_rv_nodes(nodes, spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

        # b, n, _ = s_leaves.size()
        # lengths = torch.zeros(b, n, device=s_leaves.device).int()

        src_o = {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,

            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,

            'node_indices': s_spans,
        }

        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source(left_pad_source)
    src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    # if is_infer:
    #     src_tokens = torch.cat([torch.flip(src['node_nodes'], [2]), src['node_leaves']], 2)
    #     src_labels = torch.cat([torch.flip(src['label_nodes'], [2]), src['label_leaves']], 2)
    # else:
    #     # fixme: to ensure integrity with previous codes
    #     src_tokens = torch.flip(torch.cat([src['node_leaves'], src['node_nodes']], 2), [2])
    #     src_labels = torch.flip(torch.cat([src['label_leaves'], src['label_nodes']], 2), [2])
    src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 1)
    src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 1)

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
            'src_node_leaves': src['node_leaves'],
            'src_node_nodes': src['node_nodes'],

            'src_label_leaves': src['label_leaves'],
            'src_label_nodes': src['label_nodes'],

            'src_node_indices': src['node_indices'],

            # 'src_sent_lengths': src['length'],
            'src_lengths': src_lengths,

            'src_tokens': src_tokens,
            'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class NstackMerged2SeqPairDataset(FairseqDataset):
    def __init__(
            self, srcs, src_sizes, src_dict,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True, remove_eos_from_source=False,
            append_eos_to_target=False, is_infer=False
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.srcs = srcs
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.span_data = srcs['spans']

        self.tgt = tgt
        # self.src_nsents = np.array(src_nsents)
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        assert len(self.src_sizes) == len(self.tgt_sizes), f'{len(self.src_sizes)} != {len(self.tgt_sizes)}'

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
        self.is_infer = is_infer
        same = self.tgt_dict.eos() == self.src_dict.eos()
        assert same
        # print(f'| ATTENTION ! EOS same: {same}')

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        # src_item = self.src[index]
        src_item = {k: v[index] for k, v in self.srcs.items()}

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            # if self.tgt and self.tgt[index][-1] != eos:
            #     tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
            # assert tgt_item is None or tgt_item.numel() > 0, f'{src_item}\n\n{tgt_item}'
            if tgt_item.numel() == 0:
                print(src_item)
                text_l = self.src_dict.string(src_item['leaves'])
                text_n = self.src_dict.string(src_item['nodes'])
                print(text_l)
                print(text_n)
                raise AssertionError
            if tgt_item is not None and tgt_item[-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

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
        return nstackmerge2seq_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            is_infer=self.is_infer
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
        # ['leaves', 'nodes', 'pos_tags', 'spans']
        leave_len = (src_len + 1) // 2
        node_len = src_len - leave_len

        leaves = self.src_dict.dummy_sentence(leave_len)
        pos_tags = self.src_dict.dummy_sentence(leave_len)
        nodes = self.src_dict.dummy_sentence(node_len)
        spans = torch.tensor([0, leave_len - 1]).view(1, 2).expand(node_len, 2)

        example = {
            'leaves': leaves,
            'pos_tags': pos_tags,
            'nodes': nodes,
            'spans': spans,
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
        print(f'| {self.__class__.__name__}:prefetch:starting...')
        for k, v in self.srcs.items():
            v.prefetch(indices)
            # print(f'| {self.__class__.__name__}:prefetch:{k}')
        # print(f'| {self.__class__.__name__}:prefetch:tgt')
        self.tgt.prefetch(indices)
        # print(f'| {self.__class__.__name__}:prefetch:finished...')

    @property
    def supports_prefetch(self):
        return (
                hasattr(self.src, 'supports_prefetch')
                and self.src.supports_prefetch
                and hasattr(self.tgt, 'supports_prefetch')
                and self.tgt.supports_prefetch
        )






