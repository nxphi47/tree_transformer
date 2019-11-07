import itertools
import numpy as np
import os
import torch
from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset,
    TruncatedDictionary,
)

from . import data_utils

from ..data import task_utils, monolingual_classification_dataset, MonolingualClassificationDataset, \
    DPTreeMonoClassificationDataset, \
    DPTreeIndexedCachedDataset, \
    DPTreeSeparateMonoClassificationDataset, \
    DPTreeSeparateNodeMonoClassificationDataset, \
    DPTreeSeparateLIClassificationDataset, \
    NodeStackFromDPTreeSepNodeTargetMonoClassificationDataset, \
    NodeStackFromDPTreeSepMonoClassificationDataset

from fairseq.data import data_utils, FairseqDataset, iterators, Dictionary

from fairseq import tokenizer
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math

from fairseq.tasks import FairseqTask, register_task

from .dptree2seq_translation import DPTREE_KEYS
from .fairseq_classification import SequenceClassification, MonolingualClassificationDataset

from .dptree_classification import DPTreeClassification
from .dptree_sep_classification import *


@register_task('nstack_f_dptree_classification')
class NStackFromDPTreeSeparateClassification(DPTreeSeparateClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in DPTREE_KEYS]
            if all(exists):
                return True
            else:
                print(f'Following modality not exists: {exists}')
                return False

        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert IndexedCachedDataset.exists(path), f'IndexedCachedDataset.exists({path})'
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        def dptree_indexed_dataset(path):
            assert DPTreeIndexedCachedDataset.exists(path), f'DPTreeIndexedCachedDataset.exists({path})'
            return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)

        src_datasets = []
        tgt_datasets = []
        src_datasets_dict = {k: [] for k in DPTREE_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src, tgt = 'input', 'target'

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            if split_exists(split_k, src, data_path):
                prefix = os.path.join(data_path, f'{split}.')
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
            # src_datasets.append(indexed_dataset(prefix + src))
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[DPTREE_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        src_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # print(f'src_sizes::: {src_sizes}')
        self.datasets[split] = NodeStackFromDPTreeSepMonoClassificationDataset(
            # srcs, src_sizes, src_dict
            src_dataset_dict, src_sizes, self.source_dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )


@register_task('nstack_f_dptree_node_classification')
class NStackFromDPTreeSeparateNodeClassification(DPTreeSeparateNodeClassification):

    @staticmethod
    def add_args(parser):
        DPTreeSeparateNodeClassification.add_args(parser)
        parser.add_argument('--only_binary', action='store_true', help='only_binary')
        parser.add_argument('--filter_class_index', default=2, type=int)

    def get_batch_iterator(
            self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
            ignore_invalid_inputs=False, required_batch_size_multiple=1,
            seed=1, num_shards=1, shard_id=0, num_workers=0, filter_class_index=None
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).

        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        assert isinstance(dataset, FairseqDataset)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large

        indices = data_utils.filter_by_size(
            indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        if filter_class_index is not None or self.args.only_binary:
            filter_class_index = filter_class_index if filter_class_index is not None else self.args.filter_class_index
            # print(f'Filtering data of class {filter_class_index}')
            class_fn = dataset.sample_class
            assert class_fn is not None
            # indices = task_utils.filter_by_class_size(
            #     indices, dataset.size, max_positions, class_fn, filter_class_index,
            #     raise_exception=(not ignore_invalid_inputs),
            # )
            indices = task_utils.filter_by_class(
                indices, class_fn, filter_class_index,
                raise_exception=False,
            )

        # else:
        #     indices = data_utils.filter_by_size(
        #         indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
        #     )

        # if filter_class_index is not None or self.args.only_binary:
        #     # assert isinstance(filter_class_index, int)
        #     filter_class_index = filter_class_index if filter_class_index is not None else self.args.filter_class_index
        #     print(f'Filtering data of class {filter_class_index}')
        #     class_fn = dataset.sample_class
        #     assert class_fn is not None
        #     indices = task_utils.filter_by_class(
        #         indices, class_fn, filter_class_index, raise_exception=(not ignore_invalid_inputs),
        #     )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        return iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
        )

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # assert split in ['train', 'valid', 'valid1', 'valid-bin', 'valid1-bin'], f'invalid: {split}'
        get_binary = "-bin" in split
        split_name = split
        split = split.replace('-bin', '')

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in DPTREE_KEYS]
            if all(exists):
                return True
            else:
                print(f'Following modality not exists: {exists}')
                return False

        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert IndexedCachedDataset.exists(path), f'IndexedCachedDataset.exists({path})'
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        def dptree_indexed_dataset(path):
            assert DPTreeIndexedCachedDataset.exists(path), f'DPTreeIndexedCachedDataset.exists({path})'
            return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)

        src_datasets = []
        # tgt_datasets = []
        src_datasets_dict = {k: [] for k in DPTREE_KEYS}

        data_path = self.args.data
        lang = self.args.source_lang

        src, tgt = 'txt', 'target'

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            if split_exists(split_k, src, data_path):
                prefix = os.path.join(data_path, f'{split}.')
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets_dict[DPTREE_KEYS[0]][-1])))
            if not combine:
                break

        assert len(src_datasets_dict[DPTREE_KEYS[0]]) == len(src_datasets_dict[DPTREE_KEYS[1]])

        if len(src_datasets_dict[DPTREE_KEYS[0]]) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            # tgt_dataset = tgt_datasets[0]
        else:
            # sample_ratios = [1] * len(src_datasets)
            # sample_ratios[0] = self.args.upsample_primary
            # # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            # src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            # tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            raise NotImplementedError

        src_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        self.datasets[split_name] = NodeStackFromDPTreeSepNodeTargetMonoClassificationDataset(
            # srcs, src_sizes, src_dict
            src_dataset_dict, src_sizes, self.source_dictionary,
            None,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )





def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    return tree_string

from nltk import Tree
def tree_from_string(tree_string):
    try:
        s = tree_string
        s = tree_str_post_process(s)
        tree = Tree.fromstring(s)
    except Exception as e:
        # print(f'Tree.fromstring(tree_string) failed, try to omit the post_process')
        try:
            tree = Tree.fromstring(tree_string)
        except Exception as e:
            print(f'ERROR: unable to parse the tree')
            print(tree_string)
            raise e
    return tree


def convert_flat(inf, outf, tarf):
    print(f'{inf} -> {outf}')
    with open(inf, 'r') as f:
        tree_lines = f.read().strip().split('\n')
    print(f'len = {len(tree_lines)}')
    flats = []
    targets = []
    for i, l in enumerate(tree_lines):
        try:
            tree = tree_from_string(l)
            leave_s = ' '.join(list(tree.leaves()))
        except Exception as e:
            print(f'Problem at index {i}: {l}')
        flats.append(leave_s)
        targets.append(tree.label())
    with open(outf, 'w') as f:
        f.write('\n'.join(flats))
    with open(tarf, 'w') as f:
        f.write('\n'.join(targets))




