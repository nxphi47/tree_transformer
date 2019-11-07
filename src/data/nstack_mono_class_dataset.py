import numpy as np
import torch
from fairseq import utils

from . import data_utils, FairseqDataset

from fairseq.data import monolingual_dataset, language_pair_dataset

from . import DPTreeSeparateMonoClassificationDataset
from .dptree_sep_mono_class_dataset import *
from nltk import Tree as nltkTree
from ..dptree.nstack_process import remove_single_nodeset, clean_maybe_rmnode


def split_node_leaves(x, idx, pad_idx, eos_idx, get_rv_idx=True, is_label=False):
    """

    :param x:   [t]
    :param idx: [t, 2]
    :return:
    """
    if not is_label:
        nonpad = x.ne(pad_idx) & x.ne(eos_idx)
        x = x[nonpad]
        idx = idx[nonpad]

    diff = idx[:, 0] != idx[:, 1]
    same = ~diff

    nodes = x[diff]
    leaves = x[same]

    leave_indices = idx[same]
    # assert (leave_indices[:, 0] == leave_indices[:, 1]).all()
    leave_indices = leave_indices[:, 0]
    leave_idx_out, leave_idx_order = leave_indices.sort()
    ordered_leaves = leaves.index_select(0, leave_idx_order)
    # src_lengths, sort_order = src_lengths.sort(descending=True)

    if not is_label:
        assert ordered_leaves.ne(pad_idx).all(), f'[{pad_idx}]: {ordered_leaves}, {x}'
        assert ordered_leaves.ne(eos_idx).all(), f'[{eos_idx}]: {ordered_leaves}, {x}'

    rv_nodes = torch.flip(nodes, [0])

    if get_rv_idx:
        indices = idx[diff]
        rv_indices = torch.flip(indices, [0])
    else:
        rv_indices = None
    return ordered_leaves, rv_nodes, rv_indices, diff, same


def collate_split_node_leaves(
        values, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False, get_node_idx=True, is_label=False):
    assert not left_pad
    nsent = max(v.size(0) for v in values)
    size = max(v.size(1) for v in values)
    seq_size = int((size + 1) // 2)
    node_size = size - seq_size
    assert len(values) == len(indices)

    seq_size += 1
    node_size += 1

    res_seq = values[0].new(len(values), nsent, seq_size).fill_(pad_idx)
    res_node = values[0].new(len(values), nsent, node_size).fill_(pad_idx)
    res_idx = indices[0].new(len(values), nsent, node_size, 2).fill_(0) if get_node_idx else None

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel(), f'{src.size()}, {dst.size()}'
        if move_eos_to_beginning:
            raise NotImplementedError
        else:
            dst.copy_(src)

    for i, (v, idx) in enumerate(zip(values, indices)):
        nsent = v.size(0)
        for j in range(nsent):
            leaves, rv_node, rv_idx, diff, same = split_node_leaves(
                v[j], idx[j], pad_idx, eos_idx, get_rv_idx=get_node_idx, is_label=is_label)

            seq_dest = res_seq[i, j, :leaves.size(0)]
            node_dest = res_node[i, j, node_size - rv_node.size(0):]

            copy_tensor(leaves, seq_dest)
            copy_tensor(rv_node, node_dest)

            if get_node_idx:
                idx_dest = res_idx[i, j, node_size - rv_idx.size(0):]
                copy_tensor(rv_idx, idx_dest)

    return res_seq, res_node, res_idx


def dptree2nstack_sep_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=False, target_src_label=False
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

        try:
            node_leaves, node_nodes, node_indices = collate_split_node_leaves(
                nodes, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                get_node_idx=True, is_label=False
            )
        except Exception as e:
            print(f'Collate Node Leaves:::::::')
            print(nodes)
            print(f'========================================')
            print(samples)
            raise e

        try:
            label_leaves, label_nodes, label_indices = collate_split_node_leaves(
                # labels, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                labels, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                get_node_idx=False, is_label=True
            )

        # te_label_leaves, te_label_nodes, te_label_indices = collate_split_node_leaves(
        #     # labels, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        #     labels, indices, 9, eos_idx, left_pad, move_eos_to_beginning,
        #     get_node_idx=False, is_label=True
        # )

        except Exception as e:
            print(f'Collate Label Leaves:::::::')
            print(labels)
            raise e

        # nodes = collate_token_list(nodes, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        labels = collate_token_list(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        # labels = collate_token_list(labels, 9, eos_idx, left_pad, move_eos_to_beginning)
        # indices = dptree_collate_sep_indices(indices, 0, 0, left_pad, move_eos_to_beginning)
        # length = torch.cat([x.unsqueeze_(0) for x in length], 0)
        length = data_utils.collate_tokens(length, 0, 0, False)

        src_o = {
            'node_leaves': node_leaves,
            'node_nodes': node_nodes,

            'label_leaves': label_leaves,
            'label_nodes': label_nodes,
            # 'te_label_leaves': te_label_leaves,
            # 'te_label_nodes': te_label_nodes,

            # 'indices': indices,

            'ori_labels': labels,
            # 'ori_nodes': nodes,
            # 'ori_indices': indices,

            'node_indices': node_indices,
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
        # ntokens = sum(len(s['source']['node_leaves']) + len(s['source']['node_nodes']) for s in samples)
        ntokens = sum(len(s['source']['nodes']) for s in samples)

    if target_src_label:
        target = torch.cat([src['label_leaves'], src['label_nodes']], 2)
        # source = torch.cat([src['node_leaves'], src['node_nodes']], 2)
        # node_indices = src['node_indices']
        # fl_source = torch.flip(source, [2])
        # node_indices_fl = torch.flip(node_indices, [2])
        # te_label_nodes = torch.flip(src['te_label_nodes'], [2])

        # og_target = target
        og_labels = src['ori_labels']
        # og_nodes = src['ori_nodes']
        # og_indices = src['ori_indices']

        # res = values[0].new(len(values), nsent, size).fill_(pad_idx)
        target = torch.flip(target, [2])
        src_labels = target

        og_top = og_labels[:, 0, 0]
        sl_top = src_labels[:, 0, 0]
        assert (og_top == sl_top).all()

        src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 2)
        src_tokens = torch.flip(src_tokens, [2])
    else:
        target = target

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
        },
        # 'target': torch.cat([src['label_nodes'], src['label_nodes']], 2) if target_src_label else target,
        'target': target,
    }
    if target_src_label:
        batch['net_input']['src_tokens'] = src_tokens
        batch['net_input']['src_labels'] = src_labels
    # print(f'Max_label {src_labels}')

    # sizes = {k: v.size() for k, v in batch['net_input'].items()}
    # print(f'batch-net-inputs: {sizes}')
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def dptree_to_flat_seq_sep_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False,
        input_feeding=False, target_src_label=False
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

        try:
            node_leaves, node_nodes, node_indices = collate_split_node_leaves(
                nodes, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                get_node_idx=True, is_label=False
            )
        except Exception as e:
            print(f'Collate Node Leaves:::::::')
            print(nodes)
            print(f'========================================')
            print(samples)
            raise e

        try:
            label_leaves, label_nodes, label_indices = collate_split_node_leaves(
                # labels, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                labels, indices, pad_idx, eos_idx, left_pad, move_eos_to_beginning,
                get_node_idx=False, is_label=True
            )

        except Exception as e:
            print(f'Collate Label Leaves:::::::')
            print(labels)
            raise e

        labels = collate_token_list(labels, pad_idx, eos_idx, left_pad, move_eos_to_beginning)
        length = data_utils.collate_tokens(length, 0, 0, False)

        src_o = {
            'node_leaves': node_leaves,
            'node_nodes': node_nodes,

            'label_leaves': label_leaves,
            'label_nodes': label_nodes,
            # 'te_label_leaves': te_label_leaves,
            # 'te_label_nodes': te_label_nodes,

            # 'indices': indices,

            'ori_labels': labels,
            # 'ori_nodes': nodes,
            # 'ori_indices': indices,

            'node_indices': node_indices,
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
        # ntokens = sum(len(s['source']['node_leaves']) + len(s['source']['node_nodes']) for s in samples)
        ntokens = sum(len(s['source']['nodes']) for s in samples)

    if target_src_label:
        target = torch.cat([src['label_leaves'], src['label_nodes']], 2)
        # source = torch.cat([src['node_leaves'], src['node_nodes']], 2)
        # node_indices = src['node_indices']
        # fl_source = torch.flip(source, [2])
        # node_indices_fl = torch.flip(node_indices, [2])
        # te_label_nodes = torch.flip(src['te_label_nodes'], [2])

        # og_target = target
        og_labels = src['ori_labels']
        # og_nodes = src['ori_nodes']
        # og_indices = src['ori_indices']

        # res = values[0].new(len(values), nsent, size).fill_(pad_idx)
        target = torch.flip(target, [2])
        src_labels = target

        og_top = og_labels[:, 0, 0]
        sl_top = src_labels[:, 0, 0]
        assert (og_top == sl_top).all()

        src_tokens = torch.cat([src['node_leaves'], src['node_nodes']], 2)
        src_tokens = torch.flip(src_tokens, [2])
    else:
        target = target

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
        },
        # 'target': torch.cat([src['label_nodes'], src['label_nodes']], 2) if target_src_label else target,
        'target': target,
    }
    if target_src_label:
        batch['net_input']['src_tokens'] = src_tokens
        batch['net_input']['src_labels'] = src_labels
    # print(f'Max_label {src_labels}')

    # sizes = {k: v.size() for k, v in batch['net_input'].items()}
    # print(f'batch-net-inputs: {sizes}')
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class NodeStackFromDPTreeSepMonoClassificationDataset(DPTreeSeparateMonoClassificationDataset):
    def dummy_sentence(self, dictionary, length, with_eos=True):
        t = torch.Tensor(length).uniform_(dictionary.nspecial + 1, len(dictionary)).long()
        if with_eos:
            t[-1] = dictionary.eos()
        return t

    def __len__(self):
        # assert len(self.src) == len(self.tgt), f'{len(self.src)} != {len(self.tgt)}'
        return len(self.src)

    def _get_dummy_source_example(self, src_len):
        # 'nodes', 'labels', 'indices', 'length']
        if src_len / 2 == src_len // 2:
            src_len += 1
        # nodes = self.src_dict.dummy_sentence(src_len)
        # labels = self.src_dict.dummy_sentence(src_len)
        nodes = self.dummy_sentence(self.src_dict, src_len, False)
        labels = self.dummy_sentence(self.src_dict, src_len, False)
        node_len = nodes.size()[0]
        seq_len = int((node_len + 1) // 2)  # w/o pad
        mid_node_len = node_len - seq_len

        # t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        length = torch.tensor([seq_len]).long()

        mid_node_indices = torch.tensor([0, seq_len - 1]).view(1, 1, 2).expand(1, mid_node_len, 2)
        leave_indices = torch.arange(seq_len).long().view(1, seq_len, 1).expand(1, seq_len, 2)

        indices = torch.cat([mid_node_indices, leave_indices], 1)

        example = {
            'nodes': nodes.unsqueeze_(0),
            'labels': labels.unsqueeze_(0),
            'indices': indices,
            'length': length
        }
        return example

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None

        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e

        # FIXME: have to remove EOS

        # if src_item['nodes'][-1] == self.src_dict.pad():
        #     src_item['nodes'] = src_item['nodes'][:-1]

        # if self.remove_eos_from_source:
        pad = self.src_dict.pad()
        eos = self.src_dict.eos()
        # assert (src_item['nodes'][:, -1] == pad).all() and (src_item['nodes'][:, -1] == eos).all(), f'nodes={src_item["nodes"]}'
        # if src_item['nodes'][-1] == pad:
        src_item['nodes'] = src_item['nodes'][:, :-1]
        src_item['labels'] = src_item['labels'][:, :-1]
        src_item['indices'] = src_item['indices'][:, :-1]
        # assert (src_item['nodes'][:, -1] == pad).all() and (src_item['nodes'][:, -1] == eos).all(), f'after-nodes={src_item["nodes"]}'

        # if self.remove_eos_from_source:
        #     raise NotImplementedError(
        #         f'remove_eos_from_source not supported, the tree should remove the eos already!')

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def collater(self, samples):
        return dptree2nstack_sep_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
            target_src_label=False
        )


class NodeStackFromDPTreeSepNodeTargetMonoClassificationDataset(NodeStackFromDPTreeSepMonoClassificationDataset):

    def __init__(self, srcs, src_sizes, src_dict, tgt=None, left_pad_source=False, max_source_positions=1024,
                 max_target_positions=1024, shuffle=True, input_feeding=False, remove_eos_from_source=False):
        super().__init__(srcs, src_sizes, src_dict, tgt, left_pad_source, max_source_positions, max_target_positions,
                         shuffle, input_feeding, remove_eos_from_source)
        self.final_labels = None
        self.retrieve_labels()

    def retrieve_labels(self):
        print(f'Retrieving final labels..., len={len(self.labels)}')
        assert self.labels.supports_prefetch
        self.labels.prefetch(indices=np.arange(len(self.labels)))
        self.final_labels = np.array([self.labels[i][0, 0] for i in range(len(self.labels))])
        print(f'Final labels: {len(self.final_labels)}, mean={self.final_labels.mean()}')
        self.labels.cache = None
        self.labels.cache_index = {}
        # if self.labels.data_file:
        # 	self.labels.data_file.close()

        print(f'Cleared labels dataset')

    def _get_dummy_source_example(self, src_len):
        # 'nodes', 'labels', 'indices', 'length']
        if src_len / 2 == src_len // 2:
            src_len += 1
        # nodes = self.src_dict.dummy_sentence(src_len)
        nodes = self.dummy_sentence(self.src_dict, src_len, False)
        labels = torch.ones_like(nodes)
        node_len = nodes.size()[0]
        seq_len = int((node_len + 1) // 2)  # w/o pad
        mid_node_len = node_len - seq_len

        # t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        length = torch.tensor([seq_len]).long()

        mid_node_indices = torch.tensor([0, seq_len - 1]).view(1, 1, 2).expand(1, mid_node_len, 2)
        leave_indices = torch.arange(seq_len).long().view(1, seq_len, 1).expand(1, seq_len, 2)

        indices = torch.cat([mid_node_indices, leave_indices], 1)
        # print(f'Dummy: nodes: {nodes}')

        example = {
            'nodes': nodes.unsqueeze_(0),
            'labels': labels.unsqueeze_(0),
            'indices': indices,
            'length': length
        }
        return example

    def sample_class(self, index):
        # label = self.labels[index][0, 0]
        # fixme: filter first then prefetch!
        # if self.final_labels is None:
        # 	assert len(self.labels) > 0
        # 	self.final_labels = np.array([self.labels[i] for i in range(len(self.labels))])
        # 	print(f'Final labels: {len(self.final_labels)}')
        return self.final_labels[index]

    @property
    def supports_prefetch(self):
        return getattr(self.src, 'supports_prefetch', False)

    def prefetch(self, indices):
        for k, v in self.srcs.items():
            v.prefetch(indices)

    def collater(self, samples):
        return dptree2nstack_sep_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
            target_src_label=True
        )


class FlatSeqFromDPTreeSepNodeTargetMonoClassificationDataset(NodeStackFromDPTreeSepNodeTargetMonoClassificationDataset):
    def collater(self, samples):
        return dptree_to_flat_seq_sep_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
            target_src_label=True
        )


def copy_tensor(src, dst, eos_idx, move_eos_to_beginning=False):
    assert dst.numel() == src.numel(), f'{src.size()}, {dst.size()}'
    if move_eos_to_beginning:
        assert src[-1] == eos_idx
        dst[0] = eos_idx
        dst[1:] = src[:-1]
    else:
        dst.copy_(src)
    dst.copy_(src)


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


def nstack_collate(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=False,
        target_src_label=False):
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


class NodeStackTreeMonoClassificationDataset(FairseqDataset):
    def __init__(
            self, srcs, src_sizes, src_dict,
            tgt=None,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):
        self.srcs = srcs
        self.src = srcs['leaves']

        self.leave_data = srcs['leaves']
        self.node_data = srcs['nodes']
        self.pos_tag_data = srcs['pos_tags']
        self.span_data = srcs['spans']

        self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        assert len(self.src_sizes) == len(self.src), f'{len(self.src_sizes)} = {len(self.src)}'
        for k, v in self.srcs.items():
            assert len(v) == len(self.src), f'wrong data size[{k}]: {len(v)} vs {len(self.src)}'

        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        eos = self.src_dict.eos()
        pad = self.src_dict.pad()

        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e
        non_eos = src_item['leaves'] != eos
        assert non_eos.all(), f"{src_item['leaves']}"

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported!')

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        # ;aldksd;laksd
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

    def __len__(self):
        assert len(self.src) == len(self.tgt), f'{len(self.src)} != {len(self.tgt)}'
        return len(self.src)

    def collater(self, samples):
        return nstack_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src, 'supports_prefetch', False)
                and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        for k, v in self.srcs.items():
            v.prefetch(indices)
        self.tgt.prefetch(indices)

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


def nstack_collate_treelstm(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=False,
        target_src_label=False):
    assert not left_pad_source
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    trees = [x['source']['tree'] for x in samples]
    leaves = [x['source']['leaves'] for x in samples]
    assert all(x.size(0) == 1 for x in leaves)
    id = torch.LongTensor([s['id'] for s in samples])
    ntokens = sum(x.numel() for x in leaves)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
    # target = target.index_select(0, sort_order)
    # ntokens = sum(len(s['target']) for s in samples)

    # if input_feeding:
    # 	raise ValueError(f'input_feeding should be false')
    else:
        # ntokens = sum(len(s['source']['node_leaves']) + len(s['source']['node_nodes']) for s in samples)
        ntokens = sum(len(s['source']['nodes']) for s in samples)

    assert target is not None
    # from nltk import Tree as nltkTree
    assert all(isinstance(x, nltkTree) for x in trees)
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'trees': trees,
            'inputs': leaves,
        },
        'target': target,
    }
    return batch


class NodeStackBinaryTreeLSTMMonoClassificationDataset(NodeStackTreeMonoClassificationDataset):

    def __init__(self, srcs, src_sizes, src_dict, tgt=None, left_pad_source=False, max_source_positions=1024,
                 max_target_positions=1024, shuffle=True, input_feeding=False, remove_eos_from_source=False):
        super().__init__(srcs, src_sizes, src_dict, tgt, left_pad_source, max_source_positions, max_target_positions,
                         shuffle, input_feeding, remove_eos_from_source)

        self.src_trees = None

    def _build_src_trees(self):
        eos = self.src_dict.eos()
        pad = self.src_dict.pad()
        self.src_trees = []
        from nltk import Tree
        from nltk import treetransforms

        for index in range(len(self)):
            if index % 100000 == 0:
                print(f'reconstruct tree {index}')
            # src_item = {k: v[index] for k, v in self.srcs.items()}
            leaves = self.srcs['leaves'][index]
            spans = self.srcs['spans'][index]
            # assert spans.size(0) == 1
            # assert leaves.size(0) == 1
            if spans.size(0) != 1:
                print(f'warning: reconstruct example with > 1 spans.')
            leaves = leaves[0]
            spans = spans[0]
            non_spec_leave = (leaves != eos) * (leaves != pad)
            non_spec_spans = spans[:, 0] != spans[:, 1]

            leaves = leaves[non_spec_leave]
            spans = spans[non_spec_spans]

            # if leaves[-1] == eos or leaves[-1] == pad:
            # 	leaves = leaves[:-1]

            src_len = len(leaves)
            brackets = list(map(lambda x: f'(# {x})', range(src_len)))

            try:
                for j, idx in enumerate(reversed(spans)):
                    brackets[idx[0]] = f"(# {brackets[idx[0]]}"
                    brackets[idx[1]] = f"{brackets[idx[1]]})"
            except Exception as e:
                print(brackets)
                print(spans)
                raise e

            try:
                s = " ".join(brackets)
                tree = Tree.fromstring(s)
                treetransforms.chomsky_normal_form(tree)
                tree = clean_maybe_rmnode(tree)
            except Exception as e:
                print(f'Fail building tree: {index}')
                print(f'{leaves}')
                print(f'{spans}')
                print(f'{s}')
                raise e

            self.src_trees.append(tree)
        last_example = ' '.join(str(self.src_trees[-1]).split())
        print(f'example: [{len(self.src_trees)}] {last_example}')

    def prefetch(self, indices):
        for k, v in self.srcs.items():
            v.prefetch(indices)
        self.tgt.prefetch(indices)
        # build the tree
        self._build_src_trees()

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None

        leaves = self.srcs['leaves'][index][0]
        tree = self.src_trees[index]

        eos = self.src_dict.eos()
        pad = self.src_dict.pad()
        non_spec = (leaves != eos) * (leaves != pad)
        leaves = leaves[non_spec].unsqueeze_(0)
        # if leaves[-1] == eos or leaves[-1] == pad:
        # 	leaves = leaves[:-1]

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported!')

        return {
            'id': index,
            'source': {'tree': tree, 'leaves': leaves},
            'target': tgt_item,
        }

    def _build_fake_tree(self, height):
        if height == 0:
            return '(# 3)'
        else:
            left = self._build_fake_tree(height - 1)
            right = self._build_fake_tree(height - 1)
            return f'(# {left} {right})'

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=1):
        from nltk import Tree
        """Return a dummy batch with a given number of tokens."""
        # src_len = utils.resolve_max_positions(src_len, max_positions, self.max_source_positions)
        # bsz = max(num_tokens // src_len, 1)
        bsz = 1
        # tree_string = "(10 (10 (10 10) (10 10)) (10 (10 (10 10) (10 (10 (10 (10 10) (10 10)) (10 (10 10) (10 (10 10) (10 10)))) (10 (10 10) (10 (10 (10 (10 10) (10 10)) (10 10)) (10 (10 10) (10 (10 10) (10 10))))))) (10 10)))"
        """
        (# x y)
        x: (# a b)
        y: (# a b)
        -> (# (# a b), (# a b))
        """
        height = int(np.log2(src_len))
        print(f'Dummy height: {height}, srclen={src_len}')
        tree_string = self._build_fake_tree(height)

        # tree_string = "(# (# (# (# (# (# 0) (# (# 1) (# 2))) (# (# 3) (# 4))) (# 5)) (# 6)) (# (# 7) (# (# (# 8) (# 9)) (# (# 10) (# 11)))))"
        tree = Tree.fromstring(tree_string)
        tokens = torch.LongTensor(list(map(int, tree.leaves()))).unsqueeze_(0)
        return self.collater([
            {
                'id': 0,
                'source': {'tree': tree, 'leaves': tokens},
                'target': torch.zeros(1, dtype=torch.long),
            }
            # for i in range(bsz)
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

    def collater(self, samples):
        return nstack_collate_treelstm(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )


def _build_fake_tree(height):
    if height == 0:
        return '(# 3)'
    else:
        left = _build_fake_tree(height - 1)
        right = _build_fake_tree(height - 1)
        return f'(# {left} {right})'


class NodeStackTreeNoDRootMonoClassificationDataset(NodeStackTreeMonoClassificationDataset):

    def __init__(self, srcs, src_sizes, src_dict, tgt=None, left_pad_source=False, max_source_positions=1024,
                 max_target_positions=1024, shuffle=True, input_feeding=False, remove_eos_from_source=False):
        super().__init__(srcs, src_sizes, src_dict, tgt, left_pad_source, max_source_positions, max_target_positions,
                         shuffle, input_feeding, remove_eos_from_source)
        self.rm_droot_activated = False

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        # self.leave_data = srcs['leaves']
        # self.node_data = srcs['nodes']
        # self.pos_tag_data = srcs['pos_tags']
        # self.span_data = srcs['spans']
        src_item = {}
        for k, v in self.srcs.items():
            try:
                src_item[k] = v[index][:1]
            except Exception as e:
                print(f'Access key {k}, index {index}, len={len(self.srcs[k])}')
                raise e
        spans = src_item['spans']
        nsent = spans.size(0)
        length = spans.size(1)
        # if nsent == 1:

        if length > 1 and (spans[:, 0] == spans[:, 1]).all():
            src_item['spans'] = spans[:, 1:]
            src_item['nodes'] = src_item['nodes'][:, 1:]
            if not self.rm_droot_activated:
                print(f'RM ROOT activated!')
                self.rm_droot_activated = True

        # for i in range(spans.size(0)):
        # 	if (spans[i, 0] == spans[i, 1]).all():
        # 		src_item['spans'][i] = spans[i, 1:]
        # 		src_item['nodes'][i] = src_item['nodes'][i, 1:]
        # 		if not self.rm_droot_activated:
        # 			print(f'RM ROOT activated!')
        # 			self.rm_droot_activated = True

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported!')

        return {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }


def nstack_collate_nli(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=False,
        target_src_label=False):
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
        # assert samples[0]['source'] is not None
        # _src1 = {k: [dic['source'][0][k] for dic in samples] for k in samples[0]['source']}
        _src1 = {k: [dic['source'][0][k] for dic in samples] for k in samples[0]['source'][0]}
        _src2 = {k: [dic['source'][1][k] for dic in samples] for k in samples[0]['source'][0]}

        def acquire_object(x):
            leaves = x['leaves']
            nodes = x['nodes']
            pos_tags = x['pos_tags']
            spans = x['spans']

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

        out_srcs = (acquire_object(_src1), acquire_object(_src2))
        return out_srcs

    id = torch.LongTensor([s['id'] for s in samples])
    # src = merge_source(left_pad_source)
    src1, src2 = merge_source(left_pad_source)
    # src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths1 = torch.LongTensor(
        [s['source'][0]['nodes'].numel() + s['source'][0]['leaves'].numel() for s in samples])
    src_lengths2 = torch.LongTensor(
        [s['source'][1]['nodes'].numel() + s['source'][1]['leaves'].numel() for s in samples])
    src_lengths = src_lengths1 + src_lengths2
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src1 = {k: v.index_select(0, sort_order) for k, v in src1.items()}
    src2 = {k: v.index_select(0, sort_order) for k, v in src2.items()}
    # src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    # src_tokens1 = torch.cat([src1['node_leaves'], src1['node_nodes']], 2)
    # src_tokens1 = torch.flip(src_tokens1, [2])
    # src_tokens2 = torch.cat([src2['node_leaves'], src2['node_nodes']], 2)
    # src_tokens2 = torch.flip(src_tokens2, [2])
    # src_labels1 = torch.cat([src1['label_leaves'], src1['label_nodes']], 2)
    # src_labels1 = torch.flip(src_labels1, [2])
    # src_labels2 = torch.cat([src2['label_leaves'], src2['label_nodes']], 2)
    # src_labels2 = torch.flip(src_labels2, [2])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
    else:
        ntokens = sum(len(s['source'][0]['nodes']) + len(s['source'][0]['leaves']) for s in samples)
        ntokens += sum(len(s['source'][1]['nodes']) + len(s['source'][1]['leaves']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_node_leaves1': src1['node_leaves'],
            'src_node_nodes1': src1['node_nodes'],
            'src_label_leaves1': src1['label_leaves'],
            'src_label_nodes1': src1['label_nodes'],
            'src_node_indices1': src1['node_indices'],
            'src_sent_lengths1': src1['length'],
            'src_lengths1': src_lengths1,

            'src_node_leaves2': src2['node_leaves'],
            'src_node_nodes2': src2['node_nodes'],
            'src_label_leaves2': src2['label_leaves'],
            'src_label_nodes2': src2['label_nodes'],
            'src_node_indices2': src2['node_indices'],
            'src_sent_lengths2': src2['length'],
            'src_lengths2': src_lengths2,

            # 'src_tokens': src_tokens,
            # 'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def nstack_collate_nli_concat(
        samples, pad_idx, eos_idx, left_pad_source=False, left_pad_target=False, input_feeding=False,
        target_src_label=False):
    assert not left_pad_source
    if len(samples) == 0:
        return {}

    def merge_fixed_tensors(key):
        values = [s[key] for s in samples]
        res = torch.cat(values, 0)
        # expect [B] dimension
        assert res.ndimension() == 1, f'ndim = {res.ndimension()}, not 1'
        return res

    def concat(x, y):
        # x,y:		[m, t]
        ndim = x.ndimension()
        t = max(x.size(1), y.size(1))
        # print(f'ndim = {ndim}')
        odims = [x.size(0) + x.size(0), t] + [x.size(i) for i in range(2, ndim)]
        out = x.new(*odims).fill_(pad_idx)
        out[:x.size(0), :x.size(1)] = x
        out[x.size(0):, :y.size(1)] = y
        return out

    def merge_source(left_pad, move_eos_to_beginning=False):
        # assert samples[0]['source'] is not None
        # _src1 = {k: [dic['source'][0][k] for dic in samples] for k in samples[0]['source']}
        _src1 = {k: [dic['source'][0][k] for dic in samples] for k in samples[0]['source'][0]}
        _src2 = {k: [dic['source'][1][k] for dic in samples] for k in samples[0]['source'][0]}

        # concat by sentences
        # _src = {k: [torch.cat([x, y], 0) for x, y in zip(_src1[k], _src2[k])] for k in _src1.keys()}
        _src = {k: [concat(x, y) for x, y in zip(_src1[k], _src2[k])] for k in _src1.keys()}

        # for

        # def acquire_object(x):
        leaves = _src['leaves']
        nodes = _src['nodes']
        pos_tags = _src['pos_tags']
        spans = _src['spans']

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

    # out_srcs = (acquire_object(_src1), acquire_object(_src2))
    # return out_srcs

    id = torch.LongTensor([s['id'] for s in samples])
    # src = merge_source(left_pad_source)
    src = merge_source(left_pad_source)
    # src_lengths = torch.LongTensor([s['source']['nodes'].numel() + s['source']['leaves'].numel() for s in samples])
    src_lengths1 = torch.LongTensor(
        [s['source'][0]['nodes'].numel() + s['source'][0]['leaves'].numel() for s in samples])
    src_lengths2 = torch.LongTensor(
        [s['source'][1]['nodes'].numel() + s['source'][1]['leaves'].numel() for s in samples])
    src_lengths = src_lengths1 + src_lengths2
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    # src1 = {k: v.index_select(0, sort_order) for k, v in src1.items()}
    # src2 = {k: v.index_select(0, sort_order) for k, v in src2.items()}
    src = {k: v.index_select(0, sort_order) for k, v in src.items()}

    # src_tokens1 = torch.cat([src1['node_leaves'], src1['node_nodes']], 2)
    # src_tokens1 = torch.flip(src_tokens1, [2])
    # src_tokens2 = torch.cat([src2['node_leaves'], src2['node_nodes']], 2)
    # src_tokens2 = torch.flip(src_tokens2, [2])
    # src_labels1 = torch.cat([src1['label_leaves'], src1['label_nodes']], 2)
    # src_labels1 = torch.flip(src_labels1, [2])
    # src_labels2 = torch.cat([src2['label_leaves'], src2['label_nodes']], 2)
    # src_labels2 = torch.flip(src_labels2, [2])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge_fixed_tensors('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            raise ValueError(f'input_feeding should be false')
    else:
        ntokens = sum(len(s['source'][0]['nodes']) + len(s['source'][0]['leaves']) for s in samples)
        ntokens += sum(len(s['source'][1]['nodes']) + len(s['source'][1]['leaves']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_node_leaves1': src['node_leaves'],
            'src_node_nodes1': src['node_nodes'],
            'src_label_leaves1': src['label_leaves'],
            'src_label_nodes1': src['label_nodes'],
            'src_node_indices1': src['node_indices'],
            'src_sent_lengths1': src['length'],
            'src_lengths1': src_lengths,

            # 'src_node_leaves2': src2['node_leaves'],
            # 'src_node_nodes2': src2['node_nodes'],
            # 'src_label_leaves2': src2['label_leaves'],
            # 'src_label_nodes2': src2['label_nodes'],
            # 'src_node_indices2': src2['node_indices'],
            # 'src_sent_lengths2': src2['length'],
            # 'src_lengths2': src_lengths2,

            # 'src_tokens': src_tokens,
            # 'src_labels': src_labels,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class NodeStackTreeMonoNLIClassificationDataset(FairseqDataset):
    def __init__(
            self,
            srcs1, srcs2, src_sizes1, src_sizes2,
            src_dict,
            tgt=None,
            left_pad_source=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=False, remove_eos_from_source=False
    ):

        self.srcs = srcs1
        self.srcs1 = srcs1
        self.srcs2 = srcs2

        self.src1 = srcs1['leaves']
        self.src2 = srcs2['leaves']

        # self.leave_data = srcs['leaves']
        # self.node_data = srcs['nodes']
        # self.pos_tag_data = srcs['pos_tags']
        # self.span_data = srcs['spans']

        self.tgt = tgt

        # self.src_sizes = np.array(src_sizes1)
        self.src_sizes1 = np.array(src_sizes1)
        self.src_sizes2 = np.array(src_sizes2)

        assert len(self.src_sizes1) == len(self.src1), f'{len(self.src_sizes1)} = {len(self.src1)}'
        for k, v in self.srcs1.items():
            assert len(v) == len(self.src1), f'wrong data size[{k}]: {len(v)} vs {len(self.src1)}'
        for k, v in self.srcs2.items():
            assert len(v) == len(self.src2), f'wrong data size[{k}]: {len(v)} vs {len(self.src2)}'

        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None

        src_item = {k: v[index] for k, v in self.srcs1.items()}
        src_item2 = {k: v[index] for k, v in self.srcs2.items()}

        if self.remove_eos_from_source:
            raise NotImplementedError(f'remove_eos_from_source not supported!')

        return {
            'id': index,
            'source': (src_item, src_item2),
            'target': tgt_item,
        }

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)
        # ;aldksd;laksd
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
        return indices[np.argsort(self.src_sizes1[indices] + self.src_sizes2[indices], kind='mergesort')]

    def __len__(self):
        assert len(self.src1) == len(self.tgt), f'{len(self.src1)} != {len(self.tgt)}'
        assert len(self.src1) == len(self.src2), f'{len(self.src1)} != {len(self.src2)}'
        return len(self.src1)

    def collater(self, samples):
        return nstack_collate_nli(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )

    @property
    def supports_prefetch(self):
        return (
                getattr(self.src1, 'supports_prefetch', False)
                and getattr(self.src2, 'supports_prefetch', False)
                and getattr(self.tgt, 'supports_prefetch', False)
        )

    def prefetch(self, indices):
        for k, v in self.srcs1.items():
            v.prefetch(indices)
        for k, v in self.srcs2.items():
            v.prefetch(indices)
        self.tgt.prefetch(indices)

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


class NodeStackTreeMonoNLIConcatClassificationDataset(NodeStackTreeMonoNLIClassificationDataset):
    def collater(self, samples):
        return nstack_collate_nli_concat(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            input_feeding=self.input_feeding,
        )
