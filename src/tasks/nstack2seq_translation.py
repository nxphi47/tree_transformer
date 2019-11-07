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

from ..data import DPTreeIndexedCachedDataset, Nstack2SeqPairDataset, NstackMerged2SeqPairDataset

from fairseq import tokenizer
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import FairseqDataset, iterators


# from .dptree_sep_classification import DPTreeSeparateClassification
from .dptree2seq_sep_translation import DPTree2SeqSeparateTranslationTask
from .nstack_classification import *
from ..nstack_tokenizer import NSTACK_KEYS


@register_task('nstack2seq')
class NstackTree2SeqTranslationTask(DPTree2SeqSeparateTranslationTask):

    @staticmethod
    def add_args(parser):
        DPTree2SeqSeparateTranslationTask.add_args(parser)
        parser.add_argument('--infer_mode', action='store_true', help='infer_mode')
        parser.add_argument('--get_heatmap', action='store_true', help='get_heatmap')
        parser.add_argument('--filter_nsent', default=-1, type=int)
        parser.add_argument('--on_filter_nsent', action='store_true', help='on_filter_nsent')
        parser.add_argument('--heatmap_dir', metavar='DIR', default='checkpoints', help='path to save checkpoints')

    def build_generator(self, args):
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
            raise NotImplementedError
        else:
            from ..nstack2seq_generator import Nstack2SeqGenerator, Nstack2SeqHeatmapGenerator
            assert self.target_dictionary.eos() == self.source_dictionary.eos(), f'{self.target_dictionary.eos()} - {self.source_dictionary.eos()}'
            if args.get_heatmap:
                print(f'Get {Nstack2SeqHeatmapGenerator.__name__}')
                return Nstack2SeqHeatmapGenerator(
                    args.heatmap_dir,
                    self.source_dictionary,
                    self.target_dictionary,
                    beam_size=args.beam,
                    max_len_a=args.max_len_a,
                    max_len_b=args.max_len_b,
                    min_len=args.min_len,
                    stop_early=(not args.no_early_stop),
                    normalize_scores=(not args.unnormalized),
                    len_penalty=args.lenpen,
                    unk_penalty=args.unkpen,
                    sampling=args.sampling,
                    sampling_topk=args.sampling_topk,
                    sampling_temperature=args.sampling_temperature,
                    diverse_beam_groups=args.diverse_beam_groups,
                    diverse_beam_strength=args.diverse_beam_strength,
                    match_source_len=args.match_source_len,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
            return Nstack2SeqGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def load_dataset(self, split, combine=False, **kwargs):
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                raise NotImplementedError
            elif IndexedDataset.exists(path):
                return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        tgt_datasets = []

        data_paths = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, tgt, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, tgt, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                for modality in src_datasets_dict.keys():
                    src_datasets_dict[modality].append(indexed_dataset(f'{prefix}{src}.{modality}', self.src_dict))

                # src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]

            src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            sample_ratios = [1] * len(tgt_datasets)
            sample_ratios[0] = self.args.upsample_primary
            # src_dataset = ConcatDataset(src_datasets, sample_ratios)

            src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        # src_sizes = src_dataset_dict['nodes'].sizes
        # src_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        leave_shape = src_dataset_dict['leaves'].sizes.reshape(-1, 2)
        node_shape = src_dataset_dict['nodes'].sizes.reshape(-1, 2)
        # leaves_sizes = leave_shape.sum(-1)
        # nodes_sizes = node_shape.sum(-1)
        leaves_sizes = leave_shape.prod(-1)
        nodes_sizes = node_shape.prod(-1)
        # print(f'| FIXED VERSION, size must be prod(-1)')
        src_sizes = leaves_sizes + nodes_sizes
        src_nsents = leave_shape[:, 0]
        # print(f'Some leave_size: {leave_shape[:10]}')
        # print(f'Some src_nsent: ({src_nsents[:10]})')

        self.datasets[split] = Nstack2SeqPairDataset(
            src_dataset_dict, src_sizes, self.src_dict, src_nsents,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            remove_eos_from_source=self.args.remove_eos_from_source,
            append_eos_to_target=self.args.append_eos_to_target,
            input_feeding=self.args.input_feeding,
            is_infer=self.args.infer_mode
        )

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0,
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

        size_fn = dataset.size
        if self.args.on_filter_nsent:
            print(f'| [Warning] target_max_positions ({max_positions}) used for nsent filtering')
            size_fn = dataset.src_size_nsent

        # filter examples that are too large
        indices = data_utils.filter_by_size(
            indices, size_fn, max_positions, raise_exception=(not ignore_invalid_inputs),
        )

        # if self.args.filter_nsent > 1:
        #     print(f'Filter with nsent = {self.args.filter_nsent}')
        #     filter_by_nsent()

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


@register_task('nstack_merge2seq')
class NstackMergeTree2SeqTranslationTask(DPTree2SeqSeparateTranslationTask):
    @staticmethod
    def add_args(parser):
        DPTree2SeqSeparateTranslationTask.add_args(parser)
        parser.add_argument('--infer_mode', action='store_true', help='infer_mode')
        parser.add_argument('--get_heatmap', action='store_true', help='get_heatmap')
        parser.add_argument('--filter_nsent', default=-1, type=int)
        parser.add_argument('--on_filter_nsent', action='store_true', help='on_filter_nsent')
        parser.add_argument('--heatmap_dir', metavar='DIR', default='checkpoints', help='path to save checkpoints')
        # parser.add_argument('--sample_percent', default=-1, type=float)

    # def get_batch_iterator(self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #                        ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=1, num_shards=1,
    #                        shard_id=0, num_workers=0):
    #     assert isinstance(dataset, FairseqDataset)
    #
    #     # get indices ordered by example size
    #     with data_utils.numpy_seed(seed):
    #         indices = dataset.ordered_indices()
    #
    #     # filter examples that are too large
    #     indices = data_utils.filter_by_size(
    #         indices, dataset.size, max_positions, raise_exception=(not ignore_invalid_inputs),
    #     )
    #
    #     # if self.args.sample_percent > 0:
    #     #     assert self.args.sample_percent < 1
    #     #     sample_size = int(len(indices) * self.args.sample_percent)
    #     #     print(f'| sample-percent {self.args.sample_percent}: {sample_size} / {len(indices)}')
    #     #     indices = indices[:sample_size]
    #
    #     # create mini-batches with given size constraints
    #     batch_sampler = data_utils.batch_by_size(
    #         indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
    #         required_batch_size_multiple=required_batch_size_multiple,
    #     )
    #
    #     # return a reusable, sharded iterator
    #     return iterators.EpochBatchIterator(
    #         dataset=dataset,
    #         collate_fn=dataset.collater,
    #         batch_sampler=batch_sampler,
    #         seed=seed,
    #         num_shards=num_shards,
    #         shard_id=shard_id,
    #         num_workers=num_workers,
    #     )

    def build_generator(self, args):
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
            raise NotImplementedError
        else:
            from ..nstack2seq_generator import Nstack2SeqGenerator, Nstack2SeqHeatmapGenerator
            from ..nstack2seq_generator import NstackMerge2SeqGenerator
            from ..nstack2seq_generator import NstackMerge2SeqHeatmapsGenerator
            assert self.target_dictionary.eos() == self.source_dictionary.eos(), f'{self.target_dictionary.eos()} - {self.source_dictionary.eos()}'
            if args.get_heatmap:
                print(f'| NstackMerge2SeqHeatmapsGenerator.......')
                return NstackMerge2SeqHeatmapsGenerator(
                    self.target_dictionary,
                    beam_size=args.beam,
                    max_len_a=args.max_len_a,
                    max_len_b=args.max_len_b,
                    min_len=args.min_len,
                    stop_early=(not args.no_early_stop),
                    normalize_scores=(not args.unnormalized),
                    len_penalty=args.lenpen,
                    unk_penalty=args.unkpen,
                    sampling=args.sampling,
                    sampling_topk=args.sampling_topk,
                    sampling_temperature=args.sampling_temperature,
                    diverse_beam_groups=args.diverse_beam_groups,
                    diverse_beam_strength=args.diverse_beam_strength,
                    match_source_len=args.match_source_len,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    retain_dropout=False,
                )
            print(f'| WARNING not sure if Nstack2SeqGenerator work correctly in {self.__class__.__name__}')
            return NstackMerge2SeqGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def load_dataset(self, split, combine=False, **kwargs):
        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                raise NotImplementedError
            elif IndexedDataset.exists(path):
                return DPTreeIndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets_dict = {k: [] for k in NSTACK_KEYS}
        tgt_datasets = []

        data_paths = self.args.data
        print(f'| split = {split}')
        print(f'| self.args.data = {self.args.data}')

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, tgt, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, tgt, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                for modality in src_datasets_dict.keys():
                    src_datasets_dict[modality].append(indexed_dataset(f'{prefix}{src}.{modality}', self.src_dict))

                # src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(tgt_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets_dict[NSTACK_KEYS[0]]) == len(tgt_datasets)

        if len(tgt_datasets) == 1:
            # src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
            src_dataset_dict = {k: v[0] for k, v in src_datasets_dict.items()}
            tgt_dataset = tgt_datasets[0]
        else:
            sample_ratios = [1] * len(tgt_datasets)
            sample_ratios[0] = self.args.upsample_primary
            # src_dataset = ConcatDataset(src_datasets, sample_ratios)
            src_dataset_dict = {k: ConcatDataset(v, sample_ratios) for k, v in src_datasets_dict.items()}
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        # src_sizes = src_dataset_dict['nodes'].sizes
        # src_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)
        leave_shape = src_dataset_dict['leaves'].sizes
        node_shape = src_dataset_dict['nodes'].sizes
        # leaves_sizes = leave_shape.sum(-1)
        # nodes_sizes = node_shape.sum(-1)
        leaves_sizes = leave_shape
        nodes_sizes = node_shape
        # print(f'| FIXED VERSION, size must be prod(-1)')
        src_sizes = leaves_sizes + nodes_sizes
        # src_nsents = leave_shape[:, 0]
        # print(f'Some leave_size: {leave_shape[:10]}')
        # print(f'Some src_nsent: ({src_nsents[:10]})')

        self.datasets[split] = NstackMerged2SeqPairDataset(
            src_dataset_dict, src_sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            remove_eos_from_source=self.args.remove_eos_from_source,
            append_eos_to_target=self.args.append_eos_to_target,
            input_feeding=self.args.input_feeding,
            is_infer=self.args.infer_mode
        )

import torch.nn as nn
class X(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Dropout(0.5)
    def forward(self, x):
        return self.layer(x)

m = X()
m.train()
m(torch.tensor([1.0, 2.0, 3.0, 4.0]))
m.eval()
m(torch.tensor([1.0, 2.0, 3.0, 4.0]))
