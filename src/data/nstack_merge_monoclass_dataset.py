import numpy as np
import torch
from fairseq import utils

from . import data_utils, FairseqDataset

from fairseq.data import monolingual_dataset, language_pair_dataset

from . import DPTreeSeparateMonoClassificationDataset
from .dptree_sep_mono_class_dataset import *
from nltk import Tree as nltkTree
from ..dptree.nstack_process import clean_maybe_rmnode
from ..nstack_tokenizer import PLACE_HOLDER

def copy_tensor(src, dst, eos_idx, move_eos_to_beginning=False):
    assert dst.numel() == src.numel(), f'{src.size()}, {dst.size()}'
    if move_eos_to_beginning:
        assert src[-1] == eos_idx
        dst[0] = eos_idx
        dst[1:] = src[:-1]
    else:
        dst.copy_(src)
    dst.copy_(src)


def default_collate_spans(spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    size = max(x.size(0) for x in spans)
    res_idx = spans[0].new(len(spans), size, 2).fill_(0)

    for i, v in enumerate(spans):
        # copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
        idx_dest = res_idx[i, size - len(v):]
        copy_tensor(v, idx_dest, eos_idx, move_eos_to_beginning)
    return res_idx


def nstackmerge_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True, is_infer=False,
        add_root_node=False
):
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

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        # nodes = [torchfor l, n in zip(leaves, nodes)]
        if add_root_node:
            nodes = [torch.cat((x, torch.LongTensor([eos_idx])), 0) for x in nodes]
            spans = [torch.cat((x, torch.tensor([[0, l.numel()]])), 0) for x, l in zip(spans, leaves)]

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
        # target = merge_target('target', left_pad=left_pad_target)
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


class NstackMergeMonoClassificationDataset(FairseqDataset):
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
            self, srcs, src_sizes, src_dict,
            tgt=None,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False,
            add_root_node=False,
    ):
        # if tgt_dict is not None:
        #     assert src_dict.pad() == tgt_dict.pad()
        #     assert src_dict.eos() == tgt_dict.eos()
        #     assert src_dict.unk() == tgt_dict.unk()
        self.srcs = srcs
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.span_data = srcs['spans']
        self.add_root_node = add_root_node

        self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        assert len(self.src_sizes) == len(self.src), f'{len(self.src_sizes)} = {len(self.src)}'
        for k, v in self.srcs.items():
            assert len(v) == len(self.src), f'wrong data size[{k}]: {len(v)} vs {len(self.src)}'
        # assert len(self.src_sizes) == len(self.labels), f'{len(self.src_sizes)} = {len(self.labels)}'

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
        # src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e
        # src_item = {k: v[index] for k, v in self.srcs.items()}

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def __len__(self):
        assert len(self.src) == len(self.tgt), f'{len(self.src)} != {len(self.tgt)}'
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
        return nstackmerge_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            # left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            add_root_node=self.add_root_node,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        # src_len, tgt_len = utils.resolve_max_positions(
        #     (src_len, tgt_len),
        #     max_positions,
        #     (self.max_source_positions, self.max_source_positions),
        # )
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        # bsz = max(num_tokens // src_len, 1)
        bsz = 1
        return self.collater([
            {
                'id': i,
                # 'source': self.src_dict.dummy_sentence(src_len),
                'source': self._get_dummy_source_example(src_len),
                # 'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'target': torch.zeros(1, dtype=torch.long),
            }
            for i in range(bsz)
        ])

    def _get_dummy_source_example(self, src_len):
        # ['leaves', 'nodes', 'pos_tags', 'spans']
        # leave_len = (src_len + 1) // 2
        # node_len = src_len - leave_len
        leave_len = src_len
        node_len = src_len - 1

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
        # self.src.prefetch(indices)
        for k, v in self.srcs.items():
            v.prefetch(indices)
        self.tgt.prefetch(indices)


def nstackmerge_toseq_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=True, is_infer=False):
    # assert not left_pad_source
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

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def merge_source(left_pad, move_eos_to_beginning=False):
        assert samples[0]['source'] is not None
        src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

        leaves = src['leaves']
        # tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])
        leaves = [torch.cat((x, torch.LongTensor([eos_idx])),0) for x in leaves]
        # nodes = src['nodes']
        # pos_tags = src['pos_tags']
        # spans = src['spans']

        s_leaves = data_utils.collate_tokens(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # s_pos_tags = data_utils.collate_tokens(pos_tags, pad_idx, eos_idx, False, move_eos_to_beginning)
        # s_nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, True, move_eos_to_beginning)
        # s_spans = default_collate_spans(spans, pad_idx, eos_idx, True, move_eos_to_beginning)

        # s_leaves = collate_nstack_leaves(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # s_pos_tags = collate_nstack_leaves(pos_tags, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # s_nodes, s_spans = collate_nstack_rv_nodes(nodes, spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

        # b, n, _ = s_leaves.size()
        # lengths = torch.zeros(b, n, device=s_leaves.device).int()

        src_o = {
            'node_leaves': s_leaves,
            # 'node_nodes': s_nodes,
            #
            # 'label_leaves': s_pos_tags,
            # 'label_nodes': s_nodes,
            #
            # 'node_indices': s_spans,
        }

        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source(left_pad_source)
    src_lengths = torch.LongTensor([s['source']['leaves'].numel() for s in samples])
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
    src_tokens = src['node_leaves']
    # src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 1)

    prev_output_tokens = None
    target = None

    if samples[0].get('target', None) is not None:
        # target = merge_target('target', left_pad=left_pad_target)
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
            # 'src_node_leaves': src['node_leaves'],
            # 'src_node_nodes': src['node_nodes'],
            #
            # 'src_label_leaves': src['label_leaves'],
            # 'src_label_nodes': src['label_nodes'],
            #
            # 'src_node_indices': src['node_indices'],

            # 'src_sent_lengths': src['length'],
            'src_lengths': src_lengths,

            'src_tokens': src_tokens,
            # 'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class NstackMergeToSeqMonoClassificationDataset(NstackMergeMonoClassificationDataset):

    def __getitem__(self, index):
        return super().__getitem__(index)

    def collater(self, samples):
        return nstackmerge_toseq_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )


class NstackMergeSST5MonoClassificationDataset(FairseqDataset):

    def __init__(
            self, srcs, src_sizes, src_dict,
            # tgt=None,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):
        # if tgt_dict is not None:
        #     assert src_dict.pad() == tgt_dict.pad()
        #     assert src_dict.eos() == tgt_dict.eos()
        #     assert src_dict.unk() == tgt_dict.unk()
        self.srcs = srcs
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.label_leave_data = srcs['label_leaves']
        self.label_node_data = srcs['label_nodes']

        self.span_data = srcs['spans']

        self.final_labels = None
        self.retrieve_labels()
        # self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        assert len(self.src_sizes) == len(self.src), f'{len(self.src_sizes)} = {len(self.src)}'
        for k, v in self.srcs.items():
            assert len(v) == len(self.src), f'wrong data size[{k}]: {len(v)} vs {len(self.src)}'
        # assert len(self.src_sizes) == len(self.labels), f'{len(self.src_sizes)} = {len(self.labels)}'

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

    def retrieve_labels(self):
        print(f'Retrieving final labels..., len={len(self.label_node_data)}')
        assert self.label_node_data.supports_prefetch
        self.label_node_data.prefetch(indices=np.arange(len(self.label_node_data)))
        self.final_labels = np.array([self.label_node_data[i][-1] for i in range(len(self.label_node_data))])
        print(f'Final labels: {len(self.final_labels)}, mean={self.final_labels.mean()}')
        self.label_node_data.cache = None
        self.label_node_data.cache_index = {}
        print(f'Cleared labels dataset')

    def __getitem__(self, index):
        # tgt_item = self.tgt[index] if self.tgt is not None else None
        # src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        # if self.append_eos_to_target:
        #     eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
        #     if self.tgt and self.tgt[index][-1] != eos:
        #         tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e
        # src_item = {k: v[index] for k, v in self.srcs.items()}

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': src_item,
            # 'target': tgt_item,
        }

    def __len__(self):
        return len(self.src)

    def sample_class(self, index):
        return self.final_labels[index]

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
        # return nstackmerge_ss5_collate(
        #     samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
        #     left_pad_source=self.left_pad_source,
        #     # left_pad_target=self.left_pad_target,
        #     input_feeding=self.input_feeding,
        # )
        pad_idx = self.src_dict.pad()
        eos_idx = self.src_dict.eos()
        assert not self.left_pad_source
        if len(samples) == 0:
            return {}
        if len(samples) == 0:
            return {}

        def merge_source(left_pad, move_eos_to_beginning=False):
            assert samples[0]['source'] is not None
            src = {k: [dic['source'][k] for dic in samples] for k in samples[0]['source']}

            leaves = src['leaves']
            nodes = src['nodes']
            label_leaves = src['label_leaves']
            label_nodes = src['label_nodes']

            pos_tags = src['pos_tags']
            spans = src['spans']

            s_leaves = data_utils.collate_tokens(leaves, pad_idx, eos_idx, False, move_eos_to_beginning)
            # s_pos_tags = data_utils.collate_tokens(pos_tags, pad_idx, eos_idx, False, move_eos_to_beginning)
            s_nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, True, move_eos_to_beginning)

            s_spans = default_collate_spans(spans, pad_idx, eos_idx, True, move_eos_to_beginning)

            s_label_leaves = data_utils.collate_tokens(label_leaves, 0, 0, False, move_eos_to_beginning)
            s_label_nodes = data_utils.collate_tokens(label_nodes, 0, 0, True, move_eos_to_beginning)

            assert s_leaves.size() == s_label_leaves.size(), f'{s_leaves.size()} != {s_label_leaves.size()}, {s_nodes.size()} - {s_label_nodes.size()}'
            assert s_nodes.size() == s_label_nodes.size(), f'{s_leaves.size()} != {s_label_leaves.size()}, {s_nodes.size()} - {s_label_nodes.size()}'
            # s_leaves = collate_nstack_leaves(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
            # s_pos_tags = collate_nstack_leaves(pos_tags, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
            # s_nodes, s_spans = collate_nstack_rv_nodes(nodes, spans, pad_idx, eos_idx, left_pad, move_eos_to_beginning)

            # b, n, _ = s_leaves.size()
            # lengths = torch.zeros(b, n, device=s_leaves.device).int()

            src_o = {
                'node_leaves': s_leaves,
                'node_nodes': s_nodes,

                'label_leaves': s_label_leaves,
                'label_nodes': s_label_nodes,

                'node_indices': s_spans,
            }

            return src_o

        id = torch.LongTensor([s['id'] for s in samples])
        src = merge_source(self.left_pad_source)
        src_lengths = torch.LongTensor([s['source']['leaves'].numel() + s['source']['nodes'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)

        src = {k: v.index_select(0, sort_order) for k, v in src.items()}
        src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 1)
        src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 1)
        prev_output_tokens = None
        target = src_labels

        ntokens = sum(len(s['source']['nodes']) + len(s['source']['leaves']) for s in samples)

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
        return batch

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        # src_len, tgt_len = utils.resolve_max_positions(
        #     (src_len, tgt_len),
        #     max_positions,
        #     (self.max_source_positions, self.max_source_positions),
        # )
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        # bsz = max(num_tokens // src_len, 1)
        bsz = 1
        return self.collater([
            {
                'id': i,
                # 'source': self.src_dict.dummy_sentence(src_len),
                'source': self._get_dummy_source_example(src_len),
                # 'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                # 'target': torch.zeros(1, dtype=torch.long),
            }
            for i in range(bsz)
        ])

    def _get_dummy_source_example(self, src_len):
        # ['leaves', 'nodes', 'pos_tags', 'spans']
        # leave_len = (src_len + 1) // 2
        # node_len = src_len - leave_len
        leave_len = src_len
        node_len = src_len - 1

        leaves = self.src_dict.dummy_sentence(leave_len)
        pos_tags = self.src_dict.dummy_sentence(leave_len)
        nodes = self.src_dict.dummy_sentence(node_len)
        spans = torch.tensor([0, leave_len - 1]).view(1, 2).expand(node_len, 2)

        label_leaves = torch.tensor([1]).expand(leave_len)
        label_nodes = torch.tensor([1]).expand(node_len)

        example = {
            'leaves': leaves,
            'pos_tags': pos_tags,
            'nodes': nodes,
            'spans': spans,
            'label_leaves': label_leaves,
            'label_nodes': label_nodes
        }
        return example

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
            # and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        # self.src.prefetch(indices)
        for k, v in self.srcs.items():
            v.prefetch(indices)
        # self.tgt.prefetch(indices)


def nstackmerge_relate_collate(
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

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        # assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def merge_source(key, left_pad, move_eos_to_beginning=False):
        assert samples[0][key] is not None
        src = {k: [dic[key][k] for dic in samples] for k in samples[0][key]}

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        s_leaves = data_utils.collate_tokens(leaves, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_pos_tags = data_utils.collate_tokens(pos_tags, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, True, move_eos_to_beginning)
        s_spans = default_collate_spans(spans, pad_idx, eos_idx, True, move_eos_to_beginning)

        src_o = {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,

            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,

            'node_indices': s_spans,
        }
        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source('source', left_pad_source)
    src2 = merge_source('source2', left_pad_source)
    src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src = {k: v.index_select(0, sort_order) for k, v in src.items()}
    src2 = {k: v.index_select(0, sort_order) for k, v in src2.items()}

    # src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 1)
    # src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 1)

    # target = torch.cat(values, 0)
    target = merge_fixed_tensors('target')
    tgt_score = merge_fixed_tensors('tgt_score')
    target = target.index_select(0, sort_order)
    tgt_score = tgt_score.index_select(0, sort_order)
    ntokens = sum(len(s['source']['nodes']) + len(s['source']['leaves']) for s in samples)
    ntokens += sum(len(s['source2']['nodes']) + len(s['source2']['leaves']) for s in samples)

    # print(f'target: {target}')
    # print(f'tgt_score: {tgt_score}')
    # print(f'-' * 15)

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

            'src2_node_leaves': src2['node_leaves'],
            'src2_node_nodes': src2['node_nodes'],
            'src2_label_leaves': src2['label_leaves'],
            'src2_label_nodes': src2['label_nodes'],
            'src2_node_indices': src2['node_indices'],

            'src_lengths': src_lengths,

        },
        'target': target,
        'tgt_score': tgt_score,
    }
    return batch


class NstackMergeRelateDataset(FairseqDataset):
    def __init__(
            self, srcs, src_sizes, srcs2, src2_sizes, src_dict,
            tgt, tgt_score,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False,
            nclasses=5
    ):
        # if tgt_dict is not None:
        #     assert src_dict.pad() == tgt_dict.pad()
        #     assert src_dict.eos() == tgt_dict.eos()
        #     assert src_dict.unk() == tgt_dict.unk()
        self.nclasses = nclasses
        self.srcs = srcs
        self.srcs2 = srcs2
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.span_data = srcs['spans']

        self.tgt = tgt
        self.tgt_score = tgt_score

        self.src_sizes = np.array(src_sizes)
        self.src2_sizes = np.array(src2_sizes)
        assert len(self.src_sizes) == len(self.src), f'{len(self.src_sizes)} = {len(self.src)}'
        assert len(self.src2_sizes) == len(self.src), f'{len(self.src2_sizes)} = {len(self.src)}'
        for k, v in self.srcs.items():
            assert len(v) == len(self.src), f'wrong data 1 1size[{k}]: {len(v)} vs {len(self.src)}'
        for k, v in self.srcs2.items():
            assert len(v) == len(self.src), f'wrong data 2 size[{k}]: {len(v)} vs {len(self.src)}'
        assert len(self.tgt) == len(self.tgt_score), f'{len(self.tgt_score)} vs {len(self.tgt)} vs {len(self.src)}'
        assert len(self.tgt) == len(self.src), f'{len(self.tgt_score)} vs {len(self.tgt)} vs {len(self.src)}'

        # assert len(self.src_sizes) == len(self.labels), f'{len(self.src_sizes)} = {len(self.labels)}'

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

    def __getitem__(self, index):
        tgt_item = self.tgt[index].view(1, self.nclasses)
        tgt_score = self.tgt_score[index]

        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e
        src_item = {k: v[index] for k, v in self.srcs.items()}
        src2_item = {k: v[index] for k, v in self.srcs2.items()}

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': src_item,
            'source2': src2_item,
            'target': tgt_item,
            'tgt_score': tgt_score
        }

    def __len__(self):
        assert len(self.src) == len(self.tgt), f'{len(self.src)} != {len(self.tgt)}'
        return len(self.src)

    def collater(self, samples):
        return nstackmerge_relate_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        # src_len, tgt_len = utils.resolve_max_positions(
        #     (src_len, tgt_len),
        #     max_positions,
        #     (self.max_source_positions, self.max_source_positions),
        # )
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        bsz = max(num_tokens // src_len, 1)
        bsz = 1
        return self.collater([
            {
                'id': i,
                # 'source': self.src_dict.dummy_sentence(src_len),
                'source': self._get_dummy_source_example(src_len),
                'source2': self._get_dummy_source_example(src_len),
                # 'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
                'target': torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float),
                'tgt_score': torch.tensor([2.0], dtype=torch.float),
            }
            for i in range(bsz)
        ])

    def _get_dummy_source_example(self, src_len):
        # ['leaves', 'nodes', 'pos_tags', 'spans']
        # leave_len = (src_len + 1) // 2
        # node_len = src_len - leave_len
        leave_len = src_len
        node_len = src_len - 1

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
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        for k, v in self.srcs.items():
            v.prefetch(indices)
        for k, v in self.srcs2.items():
            v.prefetch(indices)
        self.tgt.prefetch(indices)
        self.tgt_score.prefetch(indices)


def nstackmerge_relate_concat_collate(
        samples, pad_idx, eos_idx, placeholder_idx, left_pad_source=False, left_pad_target=False, input_feeding=True, is_infer=False):
    assert not left_pad_source
    if len(samples) == 0:
        return {}
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        return res

    def merge_source(key, left_pad, move_eos_to_beginning=False):
        assert samples[0][key] is not None
        src = {k: [dic[key][k] for dic in samples] for k in samples[0][key]}

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        s_leaves = data_utils.collate_tokens(leaves, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_pos_tags = data_utils.collate_tokens(pos_tags, pad_idx, eos_idx, False, move_eos_to_beginning)
        s_nodes = data_utils.collate_tokens(nodes, pad_idx, eos_idx, True, move_eos_to_beginning)
        s_spans = default_collate_spans(spans, pad_idx, eos_idx, True, move_eos_to_beginning)

        src_o = {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,

            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,

            'node_indices': s_spans,
        }
        return src_o

    def fill(x, val):
        return torch.zeros([x.numel()]).int().fill_(val)

    def merge_source_pair(key_a, key_b, left_pad, move_eos_to_beginning=False):
        assert samples[0][key_a] is not None
        assert samples[0][key_b] is not None
        src_a = {k: [dic[key_a][k] for dic in samples] for k in samples[0][key_a]}
        src_b = {k: [dic[key_b][k] for dic in samples] for k in samples[0][key_b]}

        leaves_a = src_a['leaves']
        leaves_b = src_b['leaves']

        pair = {k: [torch.cat((a, b), 0)for a, b in zip(src_a[k], src_b[k])] for k in ['leaves', 'pos_tags']}
        pair['nodes'] = [
            torch.cat((a, b, torch.tensor([placeholder_idx]).type_as(a)), 0)
            for a, b in zip(src_a['nodes'], src_b['nodes'])]
        pair['spans'] = [

            torch.cat((a, b + la.numel(), torch.tensor([[0, la.numel() + lb.numel() - 1]]).type_as(a)), 0)
            for a, b, la, lb in zip(src_a['spans'], src_b['spans'], leaves_a, leaves_b)]
        mask_le = [torch.cat((fill(x, 1), fill(y, 2)), 0) for x, y in zip(src_a['leaves'], src_b['leaves'])]
        mask_no = [torch.cat((fill(x, 1), fill(y, 2), torch.tensor([3]).int()), 0)
                   for x, y in zip(src_a['nodes'], src_b['nodes'])]

        s_leaves = data_utils.collate_tokens(pair['leaves'], pad_idx, eos_idx, False, move_eos_to_beginning)
        s_pos_tags = data_utils.collate_tokens(pair['pos_tags'], pad_idx, eos_idx, False, move_eos_to_beginning)
        s_mask_le = data_utils.collate_tokens(mask_le, 0, eos_idx, False, move_eos_to_beginning)

        s_nodes = data_utils.collate_tokens(pair['nodes'], pad_idx, eos_idx, True, move_eos_to_beginning)
        s_spans = default_collate_spans(pair['spans'], pad_idx, eos_idx, True, move_eos_to_beginning)
        s_mask_no = data_utils.collate_tokens(mask_no, 0, eos_idx, True, move_eos_to_beginning)

        assert s_mask_no.size(1) == s_nodes.size(1), f'{s_nodes.size()}, {s_mask_no.size()}, {s_spans.size()}'
        assert s_spans.size(1) == s_nodes.size(1), f'{s_nodes.size()}, {s_mask_no.size()}, {s_spans.size()}'
        assert s_mask_le.size(1) == s_leaves.size(1), f'{s_leaves.size()}, {s_mask_le.size()}, {s_spans.size()}'

        src_o = {
            'node_leaves': s_leaves,
            'node_nodes': s_nodes,

            'label_leaves': s_pos_tags,
            'label_nodes': s_nodes,

            'node_indices': s_spans,

            'mask_le': s_mask_le,
            'mask_no': s_mask_no,
        }
        return src_o

    id = torch.LongTensor([s['id'] for s in samples])
    src = merge_source_pair('source', 'source2', left_pad_source)
    src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    # src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 1)
    # src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 1)

    target = merge_fixed_tensors('target')
    tgt_score = merge_fixed_tensors('tgt_score')
    target = target.index_select(0, sort_order)
    tgt_score = tgt_score.index_select(0, sort_order)
    ntokens = sum(len(s['source']['nodes']) + len(s['source']['leaves']) for s in samples)
    ntokens += sum(len(s['source2']['nodes']) + len(s['source2']['leaves']) for s in samples)

    # print(f'target: {target}')
    # print(f'tgt_score: {tgt_score}')
    # print(f'-' * 15)

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

            'src_mask_le': src['mask_le'],
            'src_mask_no': src['mask_no'],

            'src_lengths': src_lengths,
        },
        'target': target,
        'tgt_score': tgt_score,
    }
    return batch


class NstackMergeRelateConcatDataset(NstackMergeRelateDataset):

    def __init__(self, srcs, src_sizes, srcs2, src2_sizes, src_dict, tgt, tgt_score, left_pad_source=False,
                 max_source_positions=1024, max_target_positions=1024, shuffle=True, input_feeding=False,
                 remove_eos_from_source=False, nclasses=5):
        super().__init__(srcs, src_sizes, srcs2, src2_sizes, src_dict, tgt, tgt_score, left_pad_source,
                         max_source_positions, max_target_positions, shuffle, input_feeding, remove_eos_from_source,
                         nclasses)
        print(f'Placeholder: {PLACE_HOLDER} --> {src_dict.index(PLACE_HOLDER)} vs {src_dict.unk()}')
        assert src_dict.index(PLACE_HOLDER) != src_dict.unk(), f'{self.symbols[:self.nspecial]}'

    def collater(self, samples):
        return nstackmerge_relate_concat_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            placeholder_idx=self.src_dict.index(PLACE_HOLDER),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )


class Nstack2NstackMergeMonoClassificationDataset(NstackMergeMonoClassificationDataset):
    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        """Return a dummy batch with a given number of tokens."""
        src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        bsz = 1
        return self.collater([
            {
                'id': i,
                'source': self._get_dummy_source_example(src_len),
                'target': torch.zeros(1, dtype=torch.long),
            }
            for i in range(bsz)
        ])
    #
    # def _get_dummy_source_example(self, src_len):
    #     # ['leaves', 'nodes', 'pos_tags', 'spans']
    #     # leave_len = (src_len + 1) // 2
    #     # node_len = src_len - leave_len
    #     leave_len = src_len
    #     node_len = src_len - 1
    #
    #     leaves = self.src_dict.dummy_sentence(leave_len)
    #     pos_tags = self.src_dict.dummy_sentence(leave_len)
    #     nodes = self.src_dict.dummy_sentence(node_len)
    #     spans = torch.tensor([0, leave_len - 1]).view(1, 2).expand(node_len, 2)
    #
    #     example = {
    #         'leaves': leaves,
    #         'pos_tags': pos_tags,
    #         'nodes': nodes,
    #         'spans': spans,
    #     }
    #     return example

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

    def collater(self, samples):
        return nstack2nstackmerge_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            # left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            add_root_node=self.add_root_node,
        )


def get_sep_res(values, pad_idx):
    try:
        nsent = max(v.size(0) for v in values)
        size = max(v.size(1) for v in values)
        res = values[0].new(len(values), nsent, size).fill_(pad_idx)
    except RuntimeError as e:
        print(f'values: {[v.size() for v in values]}')
        raise e
    return res, size, nsent


def collate_nstack_leaves(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    assert not move_eos_to_beginning
    res, size, nsent = get_sep_res(values, pad_idx)

    for i, v in enumerate(values):
        nsent, length = v.size()
        dest = res[i, :nsent, size - len(v[0]):] if left_pad else res[i, :nsent, :len(v[0])]
        copy_tensor(v, dest, eos_idx, move_eos_to_beginning)
    return res


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


def collate_nstack2nstackmerge_leaves(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    assert not move_eos_to_beginning
    # res, size, nsent = get_sep_res(values, pad_idx)
    nsent = max(v.size(0) for v in values)
    size = max(v.size(1) for v in values)

    for i, v in enumerate(values):
        nsent, length = v.size()
        dest = res[i, :nsent, size - len(v[0]):] if left_pad else res[i, :nsent, :len(v[0])]
        copy_tensor(v, dest, eos_idx, move_eos_to_beginning)
    return res


def nstack2nstackmerge_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=False,
        target_src_label=False, add_root_node=False,):
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

        leaves = src['leaves']
        nodes = src['nodes']
        pos_tags = src['pos_tags']
        spans = src['spans']

        s_leaves = collate_nstack2nstackmerge_leaves(leaves, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        s_pos_tags = collate_nstack2nstackmerge_leaves(pos_tags, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
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

    src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 2)
    src_tokens = torch.flip(src_tokens, [2])
    src_labels = torch.cat([src['label_leaves'], src['label_nodes']], 2)
    src_labels = torch.flip(src_labels, [2])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
    else:
        # ntokens = sum(len(s['source']['node_leaves']) + len(s['source']['node_nodes']) for s in samples)
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



