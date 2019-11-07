import itertools
import numpy as np
import os
import torch
from fairseq import options, utils
from fairseq.data import (
    data_utils, Dictionary, LanguagePairDataset, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel
import math
from ..data import (
    DPTree2SeqPairDataset, DPTreeIndexedCachedDataset, DPTree2SeqSeparatePairDataset, DPTreeWrapperDictionary
)

from .dptree2seq_translation import *
from fairseq.tasks import FairseqTask, register_task


@register_task('dptree2seq_sep')
class DPTree2SeqSeparateTranslationTask(DPTree2SeqTranslationTask):


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        assert not args.left_pad_source, f'args.left_pad_source must be False'

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        args.no_strip_node_label = getattr(args, 'no_strip_node_label', False)
        src_dict = DPTreeWrapperDictionary.load(
            os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)),
            no_strip_node_label=args.no_strip_node_label)
        tgt_dict = Dictionary.load(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] DPtree-dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def build_generator(self, args):
        if args.score_reference:
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
            raise NotImplementedError
        else:
            from ..dptree2seq_generator import DPtree2SeqSeparateGenerator
            assert self.target_dictionary.eos() == self.source_dictionary.eos(), f'{self.target_dictionary.eos()} - {self.source_dictionary.eos()}'
            return DPtree2SeqSeparateGenerator(
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

        src_datasets = []
        src_datasets_dict = {k: [] for k in DPTREE_KEYS}
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

        assert len(src_datasets_dict[DPTREE_KEYS[0]]) == len(tgt_datasets)

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
        src_sizes = src_dataset_dict['nodes'].sizes.reshape(-1, 2).sum(-1)

        self.datasets[split] = DPTree2SeqSeparatePairDataset(
            src_dataset_dict, src_sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            remove_eos_from_source=self.args.remove_eos_from_source,
            append_eos_to_target=self.args.append_eos_to_target,
            input_feeding=self.args.input_feeding,
        )




