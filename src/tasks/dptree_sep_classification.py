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
    DPTreeSeparateLIClassificationDataset

from fairseq import tokenizer
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math

from fairseq.tasks import FairseqTask, register_task

from .dptree2seq_translation import DPTREE_KEYS
from .fairseq_classification import SequenceClassification, MonolingualClassificationDataset

from .dptree_classification import DPTreeClassification


@register_task('dptree_sep_classification')
class DPTreeSeparateClassification(DPTreeClassification):
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
        self.datasets[split] = DPTreeSeparateMonoClassificationDataset(
            # srcs, src_sizes, src_dict
            src_dataset_dict, src_sizes, self.source_dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )


@register_task('dptree_sep_node_classification')
class DPTreeSeparateNodeClassification(DPTreeSeparateClassification):

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
        # tgt_datasets = []
        src_datasets_dict = {k: [] for k in DPTREE_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        # print(f'| split = {split}')
        # print(f'| self.args.data = {self.args.data}')
        # singular data path
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

            # tgt_datasets.append(indexed_dataset(prefix + tgt))

            # print('| {} {} {} examples'.format(data_path, split, len(src_datasets[-1])))
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
        self.datasets[split] = DPTreeSeparateNodeMonoClassificationDataset(
            # srcs, src_sizes, src_dict
            src_dataset_dict, src_sizes, self.source_dictionary,
            None,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )


@register_task('dptree_sep_li_classification')
class DPTreeSeparateLIClassification(DPTreeSeparateClassification):

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
        src_datasets_dict_1 = {k: [] for k in DPTREE_KEYS}
        src_datasets_dict_2 = {k: [] for k in DPTREE_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src1, src2, tgt = 'input1', 'input2', 'target'

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            if split_exists(split_k, src1, data_path):
                prefix = os.path.join(data_path, f'{split}.')
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            for modality in src_datasets_dict_1.keys():
                src_datasets_dict_1[modality].append(dptree_indexed_dataset(f'{prefix}{src1}.{modality}'))
            for modality in src_datasets_dict_2.keys():
                src_datasets_dict_2[modality].append(dptree_indexed_dataset(f'{prefix}{src2}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict_1[DPTREE_KEYS[0]]) == len(tgt_datasets)
        assert len(src_datasets_dict_2[DPTREE_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_dict_1 = {k: v[0] for k, v in src_datasets_dict_1.items()}
            src_dataset_dict_2 = {k: v[0] for k, v in src_datasets_dict_2.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            # sample_ratios = [1] * len(src_datasets)
            # sample_ratios[0] = self.args.upsample_primary
            # # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            # src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict_1.items()}
            # tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            raise NotImplementedError(f'No concatenation')

        src1_sizes = src_dataset_dict_1['nodes'].sizes.reshape(-1, 2).sum(-1)
        src2_sizes = src_dataset_dict_2['nodes'].sizes.reshape(-1, 2).sum(-1)
        # print(f'src_sizes::: {src_sizes}')
        self.datasets[split] = DPTreeSeparateLIClassificationDataset(
            # srcs, src_sizes, src_dict
            # src_dataset_dict, src_sizes, self.source_dictionary,
            src_dataset_dict_1, src1_sizes, src_dataset_dict_2, src2_sizes, self.source_dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )
