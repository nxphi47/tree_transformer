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
from fairseq import tokenizer
from ..data import (
    task_utils, monolingual_classification_dataset,
    MonolingualClassificationDataset, MonolingualNLIClassificationDataset, MonolingualNLIConcatClassificationDataset,
    MonolingualNLIConcatBertClassificationDataset, BertDictionary
)

from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math

from fairseq.tasks import FairseqTask, register_task


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


@register_task('seq_classification')
class SequenceClassification(FairseqTask):
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
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
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

        assert args.source_lang is not None
        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        dictionary = None
        output_dictionary = None
        if args.data:
            # try:
            #     dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
            # except Exception as e:
            #     dictionary = Dictionary.load(os.path.join(args.data, f'dict.{args.source_lang}.txt'))

            dictionary = try_load_dictionary(args)

            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        # dict_path = os.path.join(args.data, 'dict.txt')
        # print(f'| dict_path = {dict_path}')
        # src_dict = Dictionary.load(dict_path)
        # print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        return cls(args, dictionary, output_dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # def split_exists(split, src, tgt, lang, data_path):
        #     filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        #     if self.args.raw_text and IndexedRawTextDataset.exists(filename):
        #         return True
        #     elif not self.args.raw_text and IndexedDataset.exists(filename):
        #         return True
        #     return False

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert IndexedDataset.exists(path), f'IndexedDataset.exists({path})'
            # if self.args.raw_text:
            #     return IndexedRawTextDataset(path, dictionary)
            # elif IndexedDataset.exists(path):
            #     if self.args.lazy_load:
            #         return IndexedDataset(path, fix_lua_indexing=True)
            #     else:
            #         return IndexedCachedDataset(path, fix_lua_indexing=True)
            # return None
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        src_datasets = []
        tgt_datasets = []

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
            src_datasets.append(indexed_dataset(prefix + src))
            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split, len(src_datasets[-1])))
            if not combine:
                break

        # if not split_exists(split, )

        # # fixme: train.en.input train.en.target
        # prefix = os.path.join(data_path, f'{split}.')
        #
        # src_datasets.append(indexed_dataset(prefix + src))
        # tgt_datasets.append(indexed_dataset(prefix + tgt))
        #
        # print('| {} {} {} examples'.format(data_path, split, len(src_datasets[-1])))

        # for dk, data_path in enumerate(data_paths):
        #     for k in itertools.count():
        #         split_k = split + (str(k) if k > 0 else '')
        #
        #         # infer langcode
        #         lang = self.args.source_lang

        #         # src, tgt = self.args.source_lang, self.args.target_lang
        #         # if split_exists(split_k, src, tgt, src, data_path):
        #         #     prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        #         # elif split_exists(split_k, tgt, src, src, data_path):
        #         #     prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        #         # else:
        #         #     if k > 0 or dk > 0:
        #         #         break
        #         #     else:
        #         #         raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
        #
        #         src, tgt = 'input', 'target'
        #         # fixme: train.en.input train.en.target
        #         prefix = os.path.join(data_path, f'{split_k}.')
        #
        #         src_datasets.append(indexed_dataset(prefix + src))
        #         tgt_datasets.append(indexed_dataset(prefix + tgt))
        #
        #         print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))
        #
        #         if not combine:
        #             break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = MonolingualClassificationDataset(
            src_dataset, src_dataset.sizes, self.dictionary,
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


@register_task('seq_nli_classification')
class SequenceNLIClassification(SequenceClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        # def split_exists(split, src, tgt, lang, data_path):
        #     filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        #     if self.args.raw_text and IndexedRawTextDataset.exists(filename):
        #         return True
        #     elif not self.args.raw_text and IndexedDataset.exists(filename):
        #         return True
        #     return False

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        # def split_exists(split, data_type, data_path):
        #     filename = os.path.join(data_path, f'{split}.{data_type}')
        #     if self.args.raw_text and IndexedRawTextDataset.exists(filename):
        #         return True
        #     elif not self.args.raw_text and IndexedDataset.exists(filename):
        #         return True
        #     return False


        # def indexed_dataset(path, dictionary):
        def indexed_dataset(path):
            assert IndexedDataset.exists(path), f'IndexedDataset.exists({path})'
            # if self.args.raw_text:
            #     return IndexedRawTextDataset(path, dictionary)
            # elif IndexedDataset.exists(path):
            #     if self.args.lazy_load:
            #         return IndexedDataset(path, fix_lua_indexing=True)
            #     else:
            #         return IndexedCachedDataset(path, fix_lua_indexing=True)
            # return None
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        # src_datasets = []
        src_datasets_1 = []
        src_datasets_2 = []
        tgt_datasets = []

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
            # src_datasets.append(indexed_dataset(prefix + src))
            src_datasets_1.append(indexed_dataset(prefix + src1))
            src_datasets_2.append(indexed_dataset(prefix + src2))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split, len(src_datasets_1[-1])))
            if not combine:
                break

        assert len(src_datasets_1) == len(tgt_datasets)
        assert len(src_datasets_1) == len(src_datasets_2)

        if len(src_datasets_1) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_1, src_dataset_2, tgt_dataset = src_datasets_1[0], src_datasets_2[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets_1)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset_1 = ConcatDataset(src_datasets_1, sample_ratios)
            src_dataset_2 = ConcatDataset(src_datasets_2, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = MonolingualNLIClassificationDataset(
            src_dataset_1, src_dataset_2, src_dataset_1.sizes, src_dataset_2.sizes,
            self.dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )


@register_task('seq_nli_concat_classification')
class SequenceNLIConcatClassification(SequenceNLIClassification):
    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path):
            assert IndexedDataset.exists(path), f'IndexedDataset.exists({path})'
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        # src_datasets = []
        src_datasets_1 = []
        src_datasets_2 = []
        tgt_datasets = []

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
            # src_datasets.append(indexed_dataset(prefix + src))
            src_datasets_1.append(indexed_dataset(prefix + src1))
            src_datasets_2.append(indexed_dataset(prefix + src2))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split, len(src_datasets_1[-1])))
            if not combine:
                break

        assert len(src_datasets_1) == len(tgt_datasets)
        assert len(src_datasets_1) == len(src_datasets_2)

        if len(src_datasets_1) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_1, src_dataset_2, tgt_dataset = src_datasets_1[0], src_datasets_2[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets_1)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset_1 = ConcatDataset(src_datasets_1, sample_ratios)
            src_dataset_2 = ConcatDataset(src_datasets_2, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = MonolingualNLIConcatClassificationDataset(
            src_dataset_1, src_dataset_2, src_dataset_1.sizes, src_dataset_2.sizes,
            self.dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )


@register_task('seq_nli_concat_bert_classification')
class SequenceNLIConcatBertClassification(SequenceNLIClassification):

    @classmethod
    def try_load_dictionary(cls, args):
        try:
            dict_path = os.path.join(args.data, f'dict.txt')
            print(f'| Bertdict_path = {dict_path}')
            dictionary = BertDictionary.load(dict_path)
        except FileNotFoundError as e:
            dict_path = os.path.join(args.data, f'dict.{args.source_lang}.txt')
            print(f'| Bertdict_path = {dict_path}')
            dictionary = BertDictionary.load(dict_path)
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        print(f'| args.data = {args.data}')
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        # assert args.left_pad_source, f'Need left_pad_source True as use EOS as classifcation token'

        assert args.source_lang is not None
        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        dictionary = None
        output_dictionary = None
        if args.data:
            # try:
            #     dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
            # except Exception as e:
            #     dictionary = Dictionary.load(os.path.join(args.data, f'dict.{args.source_lang}.txt'))

            dictionary = cls.try_load_dictionary(args)

            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        if args.source_lang is None:
            args.source_lang = task_utils.infer_language_mono(args.data)

        # dict_path = os.path.join(args.data, 'dict.txt')
        # print(f'| dict_path = {dict_path}')
        # src_dict = Dictionary.load(dict_path)
        # print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        return cls(args, dictionary, output_dictionary)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        print(f'Building BERT Dictionary')
        d = BertDictionary()
        for filename in filenames:
            Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_type, data_path):
            filename = os.path.join(data_path, f'{split}.{data_type}')
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path):
            assert IndexedDataset.exists(path), f'IndexedDataset.exists({path})'
            return IndexedCachedDataset(path, fix_lua_indexing=True)

        # src_datasets = []
        src_datasets_1 = []
        src_datasets_2 = []
        tgt_datasets = []

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
            # src_datasets.append(indexed_dataset(prefix + src))
            src_datasets_1.append(indexed_dataset(prefix + src1))
            src_datasets_2.append(indexed_dataset(prefix + src2))

            tgt_datasets.append(indexed_dataset(prefix + tgt))

            print('| {} {} {} examples'.format(data_path, split, len(src_datasets_1[-1])))
            if not combine:
                break

        assert len(src_datasets_1) == len(tgt_datasets)
        assert len(src_datasets_1) == len(src_datasets_2)

        if len(src_datasets_1) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_1, src_dataset_2, tgt_dataset = src_datasets_1[0], src_datasets_2[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets_1)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset_1 = ConcatDataset(src_datasets_1, sample_ratios)
            src_dataset_2 = ConcatDataset(src_datasets_2, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = MonolingualNLIConcatBertClassificationDataset(
            src_dataset_1, src_dataset_2, src_dataset_1.sizes, src_dataset_2.sizes,
            self.dictionary,
            tgt_dataset,
            left_pad_source=self.args.left_pad_source,
            # left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            # max_target_positions=self.args.max_target_positions,
        )



