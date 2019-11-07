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

from ..data import task_utils, monolingual_classification_dataset, MonolingualClassificationDataset, \
    DPTreeMonoClassificationDataset, DPTreeIndexedCachedDataset

from fairseq import tokenizer
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math

from fairseq.tasks import FairseqTask, register_task

from .dptree2seq_translation import DPTREE_KEYS
from .fairseq_classification import SequenceClassification, MonolingualClassificationDataset



def try_load_dictionary(args):
    try:
        dict_path = os.path.join(args.data, f'dict.txt')
        print(f'| dict_path = {dict_path}')
        dictionary = Dictionary.load(dict_path)
    except FileNotFoundError as e:
        dict_path = os.path.join(args.data, f'dict.{args.source_lang}.txt')
        print(f'| dict_path = {dict_path}')
        dictionary = Dictionary.load(dict_path)
    return dictionary


@register_task('dptree_classification')
class DPTreeClassification(FairseqTask):
    """
    Model following language_modeling task.
    with target as label
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        # parser.add_argument('--sample-break-mode',
        #                     choices=['none', 'complete', 'eos'],
        #                     help='If omitted or "none", fills each sample with tokens-per-sample '
        #                          'tokens. If set to "complete", splits samples only at the end '
        #                          'of sentence, but may include multiple sentences per sample. '
        #                          'If set to "eos", includes only one sentence per sample.')
        # parser.add_argument('--tokens-per-sample', default=1024, type=int,
        #                     help='max number of tokens per sample for LM dataset')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        # parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
        #                     help='max number of tokens in the source sequence')
        parser.add_argument('--max-source-positions', default=1000000, type=int, metavar='N',
                            help='max number of tokens in the source sequence')

    def __init__(self, args, dictionary, output_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        print(f'| args.data = {args.data}')
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        # assert args.left_pad_source, f'Need left_pad_source True as use EOS as classifcation token'
        assert not args.left_pad_source, f'args.left_pad_source must be False as it the root for classification'

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
            # if self.args.raw_text and IndexedRawTextDataset.exists(filename):
            #     return True
            # elif not self.args.raw_text and IndexedDataset.exists(filename):
            #     return True
            # return False
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

            # print('| {} {} {} examples'.format(data_path, split, len(src_datasets[-1])))
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

        src_sizes = src_dataset_dict['nodes'].sizes
        self.datasets[split] = DPTreeMonoClassificationDataset(
            # srcs, src_sizes, src_dict
            src_dataset_dict, src_sizes, self.source_dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        # return (self.args.max_source_positions, self.args.max_target_positions)
        # return (self.args.max_source_positions, self.args.max_source_positions)
        return self.args.max_source_positions

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.dictionary

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

