import itertools
import numpy as np
import os
import torch
from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset,
    TruncatedDictionary, FairseqDataset, iterators
)

from . import data_utils

from ..data import (
    DPTreeIndexedCachedDataset, NodeStackTreeMonoClassificationDataset,
    FloatIndexedCachedDataset,
    NodeStackTreeMonoNLIClassificationDataset,
    NodeStackTreeMonoNLIConcatClassificationDataset,
    NodeStackBinaryTreeLSTMMonoClassificationDataset,
    NodeStackTreeNoDRootMonoClassificationDataset,
    NstackMergeMonoClassificationDataset,
    NstackMergeToSeqMonoClassificationDataset,
    NstackMergeSST5MonoClassificationDataset,
    NstackMergeRelateDataset,
    NstackMergeRelateConcatDataset,
)

from fairseq import tokenizer
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math
from ..binarization import NstackTreeMergeSST5BinarizerDataset
from ..data import task_utils
from fairseq.tasks import FairseqTask, register_task

from .dptree_sep_classification import DPTreeSeparateClassification

from ..nstack_tokenizer import NSTACK_KEYS, PLACE_HOLDER


@register_task('nstack_classification')
class NstackMonoClassification(DPTreeSeparateClassification):
    @staticmethod
    def add_args(parser):
        DPTreeSeparateClassification.add_args(parser)
        parser.add_argument('--measure_size_leaves', action='store_true', help='measure_size_leaves')
        parser.add_argument('--add_root_node', action='store_true', help='add_root_node')

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes
        #
        # leaves_sizes = src_dataset_dict['leaves'].sizes
        # nodes_sizes = src_dataset_dict['nodes'].sizes
        if self.args.measure_size_leaves:
            # src_sizes = leaves_sizes + nodes_sizes
            print(f'measure_size_leaves.....')
            src_sizes = leaves_sizes
        else:
            src_sizes = leaves_sizes + nodes_sizes
        print(f'| add_root_node: {self.args.add_root_node}')

        self.datasets[split] = NodeStackTreeMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstack_nodroot_classification')
class NstackMonoNoDRootClassification(DPTreeSeparateClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        src_sizes = leaves_sizes + nodes_sizes

        self.datasets[split] = NodeStackTreeNoDRootMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstack_nli_classification')
class NstackMonoNLIClassification(NstackMonoClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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

        # src_datasets = []
        tgt_datasets = []
        # src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_1 = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_2 = {k: [] for k in NSTACK_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src, tgt = 'input', 'target'
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

            # for modality in src_datasets_dict.keys():
            # 	src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))
            for modality in src_datasets_dict_1.keys():
                src_datasets_dict_1[modality].append(dptree_indexed_dataset(f'{prefix}{src1}.{modality}'))
            for modality in src_datasets_dict_2.keys():
                src_datasets_dict_2[modality].append(dptree_indexed_dataset(f'{prefix}{src2}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict_1[NSTACK_KEYS[0]]) == len(tgt_datasets)
        assert len(src_datasets_dict_2[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            # src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            src_dataset_dict_1 = {k: v[0] for k, v in src_datasets_dict_1.items()}
            src_dataset_dict_2 = {k: v[0] for k, v in src_datasets_dict_2.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            # sample_ratios = [1] * len(src_datasets)
            # sample_ratios[0] = self.args.upsample_primary
            # # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            # src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            # tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            raise NotImplementedError(f'No concatenation')

        def get_size(_dict, keys):
            size = None
            for k in keys:
                ksize = _dict[k].sizes.reshape(-1, 2).sum(-1)
                size = ksize if size is None else size + ksize
            return size

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes
        #
        # src1_sizes = src_dataset_dict_1['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src2_sizes = src_dataset_dict_2['nodes'].sizes.reshape(-1, 2).sum(-1)

        src1_sizes = get_size(src_dataset_dict_1, ['leaves', 'nodes'])
        src2_sizes = get_size(src_dataset_dict_2, ['leaves', 'nodes'])

        self.datasets[split] = NodeStackTreeMonoNLIClassificationDataset(
            src_dataset_dict_1, src_dataset_dict_2,
            src1_sizes, src2_sizes,
            self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstack_nli_concat_lassification')
class NstackMonoNLIConcatClassification(NstackMonoNLIClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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

        # src_datasets = []
        tgt_datasets = []
        # src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_1 = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_2 = {k: [] for k in NSTACK_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src, tgt = 'input', 'target'
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

            # for modality in src_datasets_dict.keys():
            # 	src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))
            for modality in src_datasets_dict_1.keys():
                src_datasets_dict_1[modality].append(dptree_indexed_dataset(f'{prefix}{src1}.{modality}'))
            for modality in src_datasets_dict_2.keys():
                src_datasets_dict_2[modality].append(dptree_indexed_dataset(f'{prefix}{src2}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict_1[NSTACK_KEYS[0]]) == len(tgt_datasets)
        assert len(src_datasets_dict_2[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            # src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            src_dataset_dict_1 = {k: v[0] for k, v in src_datasets_dict_1.items()}
            src_dataset_dict_2 = {k: v[0] for k, v in src_datasets_dict_2.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            # sample_ratios = [1] * len(src_datasets)
            # sample_ratios[0] = self.args.upsample_primary
            # # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            # src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            # tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            raise NotImplementedError(f'No concatenation')

        def get_size(_dict, keys):
            size = None
            for k in keys:
                ksize = _dict[k].sizes.reshape(-1, 2).sum(-1)
                size = ksize if size is None else size + ksize
            return size

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes
        #
        # src1_sizes = src_dataset_dict_1['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src2_sizes = src_dataset_dict_2['nodes'].sizes.reshape(-1, 2).sum(-1)

        src1_sizes = get_size(src_dataset_dict_1, ['leaves', 'nodes'])
        src2_sizes = get_size(src_dataset_dict_2, ['leaves', 'nodes'])

        self.datasets[split] = NodeStackTreeMonoNLIConcatClassificationDataset(
            src_dataset_dict_1, src_dataset_dict_2,
            src1_sizes, src2_sizes,
            self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstack2treelstm_2nary_classification')
class Nstack2TreeLSTM2NaryMonoClassification(NstackMonoClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        src_sizes = leaves_sizes + nodes_sizes

        self.datasets[split] = NodeStackBinaryTreeLSTMMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstackmerge_classification')
class NstackMergeMonoClassification(DPTreeSeparateClassification):

    @staticmethod
    def add_args(parser):
        DPTreeSeparateClassification.add_args(parser)
        parser.add_argument('--measure_size_leaves', action='store_true', help='measure_size_leaves')
        parser.add_argument('--add_root_node', action='store_true', help='add_root_node')

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        print(f'| args.data = {args.data}')
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        # assert args.left_pad_source, f'Need left_pad_source True as use EOS as classifcation token'
        # assert not args.left_pad_source, f'args.left_pad_source must be True to take the last'

        assert args.source_lang is not None
        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        dict_path = os.path.join(args.data, 'dict.txt')
        if not os.path.exists(dict_path):
            dict_path = os.path.join(args.data, f'dict.{args.source_lang}.txt')

        dictionary = None
        output_dictionary = None
        if args.data:
            dictionary = Dictionary.load(dict_path)
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        # dict_path = os.path.join(args.data, 'dict.txt')
        # src_dict = Dictionary.load(dict_path)
        # print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        return cls(args, dictionary, output_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes

        leaves_sizes = src_dataset_dict['leaves'].sizes
        nodes_sizes = src_dataset_dict['nodes'].sizes
        if self.args.measure_size_leaves:
        # src_sizes = leaves_sizes + nodes_sizes
            print(f'measure_size_leaves.....')
            src_sizes = leaves_sizes
        else:
            src_sizes = leaves_sizes + nodes_sizes
        print(f'| add_root_node: {self.args.add_root_node}')
        # assert self.args.left_pad_source
        self.datasets[split] = NstackMergeMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
            add_root_node=self.args.add_root_node
        )

    # def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
    #     model.train()
    #     # print(f'Start forward')
    #     loss, sample_size, logging_output = criterion(model, sample)
    #     if ignore_grad:
    #         loss *= 0
    #     # print(f'Start backward')
    #     optimizer.backward(loss)
    #     # print(f'End backward')
    #     return loss, sample_size, logging_output


@register_task('nstack2nstackmerge_classification')
class Nstack2NstackMergeMonoClassification(NstackMergeMonoClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes
        #
        # leaves_sizes = src_dataset_dict['leaves'].sizes
        # nodes_sizes = src_dataset_dict['nodes'].sizes
        if self.args.measure_size_leaves:
            # src_sizes = leaves_sizes + nodes_sizes
            print(f'measure_size_leaves.....')
            src_sizes = leaves_sizes
        else:
            src_sizes = leaves_sizes + nodes_sizes
        print(f'| add_root_node: {self.args.add_root_node}')

        self.datasets[split] = NodeStack2NstackMergeTreeMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
            add_root_node=self.args.add_root_node
        )


@register_task('nstackmerge_sst5_classification')
class NstackMergeSST5MonoClassification(NstackMergeMonoClassification):
    @staticmethod
    def add_args(parser):
        NstackMergeMonoClassification.add_args(parser)
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
            class_fn = dataset.sample_class
            assert class_fn is not None
            indices = task_utils.filter_by_class(
                indices, class_fn, filter_class_index,
                raise_exception=False,
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        # asdasd
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

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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

        src_keys = NstackTreeMergeSST5BinarizerDataset.SST5_NSTACK_KEYS

        src_datasets = []
        # tgt_datasets = []
        src_datasets_dict = {k: [] for k in src_keys}

        # data_paths = self.args.data
        data_path = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        # src, tgt = 'phrase', 'target'
        src = 'phrase'

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

            print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets_dict[src_keys[0]][-1])))
            if not combine:
                break

        # assert len(src_datasets_dict[src_keys[0]]) == len(tgt_datasets)
        first_datasets = src_datasets_dict[src_keys[0]]
        if len(first_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            # tgt_dataset = tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            # tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes

        # leaves_sizes = src_dataset_dict['leaves'].sizes
        # nodes_sizes = src_dataset_dict['nodes'].sizes
        # src_sizes = leaves_sizes + nodes_sizes

        leaves_sizes = src_dataset_dict['leaves'].sizes
        nodes_sizes = src_dataset_dict['nodes'].sizes
        if self.args.measure_size_leaves:
            # src_sizes = leaves_sizes + nodes_sizes
            print(f'measure_size_leaves.....')
            src_sizes = leaves_sizes
        else:
            src_sizes = leaves_sizes + nodes_sizes
        print(f'| add_root_node: {self.args.add_root_node}')

        # assert self.args.left_pad_source
        self.datasets[split] = NstackMergeSST5MonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


@register_task('nstackmerge_relateness')
class NstackMergeRelateness(NstackMergeMonoClassification):

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
            if all(exists):
                return True
            else:
                print(f'Following modality not exists: {exists}')
                return False

        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert FloatIndexedCachedDataset.exists(path), f'IndexedCachedDataset.exists({path})'
            return FloatIndexedCachedDataset(path, fix_lua_indexing=True)

        def dptree_indexed_dataset(path):
            assert DPTreeIndexedCachedDataset.exists(path), f'DPTreeIndexedCachedDataset.exists({path})'
            return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)

        # src_datasets = []
        tgt_datasets = []
        tgt_score_datasets = []
        # src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_1 = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_2 = {k: [] for k in NSTACK_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        nclass = getattr(self.args, 'nclasses', 5)
        print(f'| split = {split}')
        print(f'| nclass = {nclass}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src, tgt = 'input', 'target'
        src1, src2, tgt, tgt_score = 'a', 'b', 'target', 'tgt_score'

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            if split_exists(split_k, src1, data_path):
                prefix = os.path.join(data_path, f'{split}.')
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

            # for modality in src_datasets_dict.keys():
            # 	src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))
            for modality in src_datasets_dict_1.keys():
                src_datasets_dict_1[modality].append(dptree_indexed_dataset(f'{prefix}{src1}.{modality}'))
            for modality in src_datasets_dict_2.keys():
                src_datasets_dict_2[modality].append(dptree_indexed_dataset(f'{prefix}{src2}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))
            tgt_score_datasets.append(indexed_dataset(prefix + tgt_score))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict_1[NSTACK_KEYS[0]]) == len(tgt_datasets)
        assert len(src_datasets_dict_2[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            # src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            src_dataset_dict_1 = {k: v[0] for k, v in src_datasets_dict_1.items()}
            src_dataset_dict_2 = {k: v[0] for k, v in src_datasets_dict_2.items()}
            tgt_dataset = tgt_datasets[0]
            tgt_score_dataset = tgt_score_datasets[0]
        else:
            raise NotImplementedError(f'No concatenation')

        def get_size(_dict, keys):
            size = None
            for k in keys:
                ksize = _dict[k].sizes
                size = ksize if size is None else size + ksize
            return size

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes
        #
        # src1_sizes = src_dataset_dict_1['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src2_sizes = src_dataset_dict_2['nodes'].sizes.reshape(-1, 2).sum(-1)

        # leaves_sizes = src_dataset_dict['leaves'].sizes
        # nodes_sizes = src_dataset_dict['nodes'].sizes
        # src_sizes = leaves_sizes + nodes_sizes

        src1_sizes = get_size(src_dataset_dict_1, ['leaves', 'nodes'])
        src2_sizes = get_size(src_dataset_dict_2, ['leaves', 'nodes'])

        self.datasets[split] = NstackMergeRelateDataset(
            src_dataset_dict_1, src1_sizes, src_dataset_dict_2, src2_sizes,
            self.source_dictionary, tgt_dataset, tgt_score_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
            nclasses=nclass
        )


@register_task('nstackmerge_toseq_classification')
class NstackMergeToSeqMonoClassification(NstackMergeMonoClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
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
        src_datasets_dict = {k: [] for k in NSTACK_KEYS}

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
            for modality in src_datasets_dict.keys():
                src_datasets_dict[modality].append(dptree_indexed_dataset(f'{prefix}{src}.{modality}'))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

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

        # leaves_sizes = src_dataset_dict['leaves'].sizes.reshape(-1, 2).sum(-1)
        # nodes_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        # src_sizes = leaves_sizes + nodes_sizes

        leaves_sizes = src_dataset_dict['leaves'].sizes
        nodes_sizes = src_dataset_dict['nodes'].sizes
        if self.args.measure_size_leaves:
        # src_sizes = leaves_sizes + nodes_sizes
            print(f'measure_size_leaves.....')
            src_sizes = leaves_sizes
        else:
            src_sizes = leaves_sizes + nodes_sizes

        # assert self.args.left_pad_source
        self.datasets[split] = NstackMergeToSeqMonoClassificationDataset(
            src_dataset_dict, src_sizes, self.source_dictionary, tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
        )


class PlholderDictionary(Dictionary):

    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls()
        d.add_symbol(PLACE_HOLDER)
        d.nspecial = len(d.symbols)

        lines = f.readlines()
        indices_start_line = d._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d


@register_task('nstackmerge_relateness_concat')
class NstackMergeRelatenessConcat(NstackMergeMonoClassification):

    @classmethod
    def setup_task(cls, args, **kwargs):
        print(f'| args.data = {args.data}')
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        # assert args.left_pad_source, f'Need left_pad_source True as use EOS as classifcation token'
        assert not args.left_pad_source, f'args.left_pad_source must be True to take the last'

        assert args.source_lang is not None
        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        dict_path = os.path.join(args.data, 'dict.txt')
        if not os.path.exists(dict_path):
            dict_path = os.path.join(args.data, f'dict.{args.source_lang}.txt')

        dictionary = None
        output_dictionary = None
        # src_dict.add_symbol(PLACE_HOLDER)
        # src_dict.nspecial = len(src_dict.symbols)

        if args.data:
            dictionary = PlholderDictionary.load(dict_path)
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                assert ValueError
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        # dict_path = os.path.join(args.data, 'dict.txt')
        # src_dict = Dictionary.load(dict_path)
        # print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        return cls(args, dictionary, output_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            assert not self.args.raw_text
            exists = [IndexedDataset.exists(os.path.join(data_path, f'{split}.{data_type}.{k}')) for k in NSTACK_KEYS]
            if all(exists):
                return True
            else:
                print(f'Following modality not exists: {exists}')
                return False

        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert FloatIndexedCachedDataset.exists(path), f'IndexedCachedDataset.exists({path})'
            return FloatIndexedCachedDataset(path, fix_lua_indexing=True)

        def dptree_indexed_dataset(path):
            assert DPTreeIndexedCachedDataset.exists(path), f'DPTreeIndexedCachedDataset.exists({path})'
            return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)

        # src_datasets = []
        tgt_datasets = []
        tgt_score_datasets = []
        # src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_1 = {k: [] for k in NSTACK_KEYS}
        src_datasets_dict_2 = {k: [] for k in NSTACK_KEYS}

        # data_paths = self.args.data
        data_path = self.args.data
        nclass = getattr(self.args, 'nclasses', 5)
        print(f'| split = {split}')
        print(f'| nclass = {nclass}')
        print(f'| self.args.data = {self.args.data}')
        # singular data path
        lang = self.args.source_lang

        src, tgt = 'input', 'target'
        src1, src2, tgt, tgt_score = 'a', 'b', 'target', 'tgt_score'

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
            tgt_score_datasets.append(indexed_dataset(prefix + tgt_score))

            print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))
            if not combine:
                break

        assert len(src_datasets_dict_1[NSTACK_KEYS[0]]) == len(tgt_datasets)
        assert len(src_datasets_dict_2[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            src_dataset_dict_1 = {k: v[0] for k, v in src_datasets_dict_1.items()}
            src_dataset_dict_2 = {k: v[0] for k, v in src_datasets_dict_2.items()}
            tgt_dataset = tgt_datasets[0]
            tgt_score_dataset = tgt_score_datasets[0]
        else:
            raise NotImplementedError(f'No concatenation')

        def get_size(_dict, keys):
            size = None
            for k in keys:
                ksize = _dict[k].sizes
                size = ksize if size is None else size + ksize
            return size

        src1_sizes = get_size(src_dataset_dict_1, ['leaves', 'nodes'])
        src2_sizes = get_size(src_dataset_dict_2, ['leaves', 'nodes'])

        self.datasets[split] = NstackMergeRelateConcatDataset(
            src_dataset_dict_1, src1_sizes, src_dataset_dict_2, src2_sizes,
            self.source_dictionary, tgt_dataset, tgt_score_dataset,
            left_pad_source=self.args.left_pad_source,
            max_source_positions=self.args.max_source_positions,
            nclasses=nclass
        )



