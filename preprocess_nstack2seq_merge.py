from collections import Counter
from itertools import zip_longest

from fairseq import options, tasks
from fairseq.data import indexed_dataset, Dictionary
from fairseq.binarizer import Binarizer
from fairseq.utils import import_user_module
from multiprocessing import Pool

from fairseq import options, tasks
from fairseq.data import indexed_dataset, dictionary
from fairseq.binarizer import Binarizer
from fairseq.utils import import_user_module


import torch
import os
import shutil
from fairseq.tokenizer import tokenize_line
from . import src

import argparse
from collections import Counter
from itertools import zip_longest
import os
import shutil
from fairseq.tokenizer import tokenize_line
import glob
from fairseq.data import dictionary
from fairseq.tokenizer import tokenize_line
from multiprocessing import Pool, Manager, Process
import datetime
from . import src
import time
from shutil import copyfile


from fairseq import options, tasks
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
from fairseq.utils import import_user_module


# DPTreeTokenizer = src.dptree_tokenizer.DPTreeTokenizer

CoreNLPTreeBuilder = src.dptree.CoreNLPTreeBuilder
# DPTREE_KEYS = src.tasks.DPTREE_KEYS

BinarizerDataset = src.binarization.BinarizerDataset
ClassBinarizerDataset = src.binarization.ClassBinarizerDataset
# DPTreeSeparateBinarizerDataset = src.binarization.DPTreeSeparateBinarizerDataset



NstackTreeBuilder = src.dptree.NstackTreeBuilder
NStackDataset = src.dptree.NStackDataset
NstackTreeTokenizer = src.nstack_tokenizer.NstackTreeTokenizer
NSTACK_KEYS = src.nstack_tokenizer.NSTACK_KEYS

# NstackTreeSeparateBinarizerDataset = src.binarization.NstackTreeSeparateBinarizerDataset
NstackTreeMergeBinarizerDataset = src.binarization.NstackTreeMergeBinarizerDataset
NstackSeparateIndexedDatasetBuilder = src.data.NstackSeparateIndexedDatasetBuilder

# !/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

# from src.dptree_tokenizer import DPTreeTokenizer

# DPTreeTokenizer = src.dptree_tokenizer.DPTreeTokenizer
# DPTreeSeparateBinarizerDataset = src.dptree_tokenizer.DPTreeSeparateBinarizerDataset
# BinarizerDataset = src.dptree_tokenizer.BinarizerDataset
# DPTreeSeparateBinarizerDataset = src.binarization.DPTreeSeparateBinarizerDataset

# DPTreeSeparateIndexedDatasetBuilder = src.data.DPTreeSeparateIndexedDatasetBuilder

# from multiprocessing import Pool


def get_parser():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("-s", "--source-lang", default=None, metavar="SRC",
                        help="source language")
    parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                        help="target language")
    parser.add_argument("--trainpref", metavar="FP", default=None,
                        help="train file prefix")
    parser.add_argument("--validpref", metavar="FP", default=None,
                        help="comma separated, valid file prefixes")
    parser.add_argument("--testpref", metavar="FP", default=None,
                        help="comma separated, test file prefixes")
    parser.add_argument("--destdir", metavar="DIR", default="data-bin",
                        help="destination dir")
    parser.add_argument("--thresholdtgt", metavar="N", default=0, type=int,
                        help="map words appearing less than threshold times to unknown")
    parser.add_argument("--thresholdsrc", metavar="N", default=0, type=int,
                        help="map words appearing less than threshold times to unknown")
    parser.add_argument("--tgtdict", metavar="FP",
                        help="reuse given target dictionary")
    parser.add_argument("--srcdict", metavar="FP",
                        help="reuse given source dictionary")
    parser.add_argument("--share_dict_txt", metavar="FP",
                        help="reuse given source dictionary")
    parser.add_argument("--nwordstgt", metavar="N", default=-1, type=int,
                        help="number of target words to retain")
    parser.add_argument("--nwordssrc", metavar="N", default=-1, type=int,
                        help="number of source words to retain")
    parser.add_argument("--alignfile", metavar="ALIGN", default=None,
                        help="an alignment file (optional)")
    parser.add_argument("--output-format", metavar="FORMAT", default="binary",
                        choices=["binary", "raw"],
                        help="output format (optional)")
    parser.add_argument("--joined-dictionary", action="store_true",
                        help="Generate joined dictionary")
    parser.add_argument("--only-source", action="store_true",
                        help="Only process the source language")
    parser.add_argument("--padding-factor", metavar="N", default=8, type=int,
                        help="Pad dictionary size to be multiple of N")
    parser.add_argument("--workers", metavar="N", default=1, type=int,
                        help="number of parallel workers")
    # fmt: on
    return parser


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_with_load(args, filename, dict_path, output_prefix, lang, offset, end):
    dict = dictionary.Dictionary.load(dict_path)
    binarize(args, filename, dict, output_prefix, lang, offset, end)
    return dataset_dest_prefix(args, output_prefix, lang)


def dataset_dest_prefix(args, output_prefix, lang):
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return f"{base}.{extension}"


# def get_offsets(input_file, num_workers):
#     return Tokenizer.find_offsets(input_file, num_workers)


def merge_files(files, outpath):
    ds = indexed_dataset.IndexedDatasetBuilder("{}.bin".format(outpath))
    for file in files:
        ds.merge_file_(file)
        os.remove(indexed_dataset.data_file_path(file))
        os.remove(indexed_dataset.index_file_path(file))
    ds.finalize("{}.idx".format(outpath))


# TODO:-------------------- dptree2seq functions!-------------------------------------


def dataset_dest_prefix_dptree(args, output_prefix, lang, modality):
    assert lang is not None
    base = f"{args.destdir}/{output_prefix}"
    lang_part = (
        f".{args.source_lang}-{args.target_lang}.{lang}.{modality}" if lang is not None else ""
    )
    return f"{base}{lang_part}"


def dataset_dest_file_dptree(args, output_prefix, lang, extension, modality):
    base = dataset_dest_prefix_dptree(args, output_prefix, lang, modality)
    return f"{base}.{extension}"



def main(args):
    import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    # group.add_argument("--convert_raw", action="store_true", help="convert_raw")
    # group.add_argument("--convert_with_bpe", action="store_true", help="convert_with_bpe")
    # group.add_argument('--bpe_code', metavar='FILE', help='bpe_code')

    # new_prefix, src_tree_file, tgt_tree_file

    if args.convert_raw:
        print(f'start --- args.convert_raw')
        raise NotImplementedError

    if args.convert_raw_only:
        print(f'Finish!.')
        return

    remove_root = not args.no_remove_root
    take_pos_tag = not args.no_take_pos_tag
    take_nodes = not args.no_take_nodes
    reverse_node = not args.no_reverse_node
    no_collapse = args.no_collapse
    # remove_root =, take_pos_tag =, take_nodes =
    print(f'remove_root: {remove_root}')
    print(f'take_pos_tag: {take_pos_tag}')
    print(f'take_nodes: {take_nodes}')
    print(f'reverse_node: {reverse_node}')
    print(f'no_collapse: {no_collapse}')

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def share_dict_path():
        return args.share_dict_txt

    def build_shared_nstack2seq_dictionary(_src_file, _tgt_file):
        d = dictionary.Dictionary()
        print(f'Build dict on src_file: {_src_file}')
        NstackTreeTokenizer.acquire_vocab_multithread(
            _src_file, d, tokenize_line, num_workers=args.workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
            no_collapse=no_collapse,
        )
        print(f'Build dict on tgt_file: {_tgt_file}')
        dictionary.Dictionary.add_file_to_dictionary(_tgt_file, d, tokenize_line, num_workers=args.workers)
        d.finalize(
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor
        )
        print(f'Finish building vocabulary: size {len(d)}')
        return d

    def build_nstack_source_dictionary(_src_file):
        d = dictionary.Dictionary()
        print(f'Build dict on src_file: {_src_file}')
        NstackTreeTokenizer.acquire_vocab_multithread(
            _src_file, d, tokenize_line, num_workers=args.workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
            no_collapse=no_collapse,
        )
        d.finalize(
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor
        )
        print(f'Finish building src vocabulary: size {len(d)}')
        return d

    def build_target_dictionary(_tgt_file):
        # assert src ^ tgt
        print(f'Build dict on tgt: {_tgt_file}')
        d = task.build_dictionary(
            [_tgt_file],
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )
        print(f'Finish building tgt vocabulary: size {len(d)}')
        return d

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    # if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
    #     raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_file = f'{args.trainpref}.{args.source_lang}'
            tgt_file = f'{args.trainpref}.{args.target_lang}'
            src_dict = build_shared_nstack2seq_dictionary(src_file, tgt_file)
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_nstack_source_dictionary(train_path(args.source_lang))

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_target_dictionary(train_path(args.target_lang))
        else:
            tgt_dict = None
        # raise NotImplementedError(f'only allow args.joined_dictionary for now')

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        pool = None

        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )

        def consumer(tensor):
            ds.add_item(tensor)

        stat = BinarizerDataset.export_binarized_dataset(
            input_file, vocab, consumer, add_if_not_exist=False, num_workers=num_workers,
        )

        ntok = stat['ntok']
        nseq = stat['nseq']
        nunk = stat['nunk']

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                nseq,
                ntok,
                100 * nunk / ntok,
                vocab.unk_word,
            )
        )

    def make_binary_nstack_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        print("| [{}] Dictionary: {} types".format(lang, len(vocab) - 1))

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )

        dss = {
            modality: NstackSeparateIndexedDatasetBuilder(
                dataset_dest_file_dptree(args, output_prefix, lang, 'bin', modality))
            for modality in NSTACK_KEYS
        }

        def consumer(example):
            for modality, tensor in example.items():
                dss[modality].add_item(tensor)

        stat = NstackTreeMergeBinarizerDataset.export_binarized_separate_dataset(
            input_file, vocab, consumer, add_if_not_exist=False, num_workers=num_workers,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node,
            no_collapse=no_collapse,
        )
        ntok = stat['ntok']
        nseq = stat['nseq']
        nunk = stat['nunk']

        for modality, ds in dss.items():
            ds.finalize(dataset_dest_file_dptree(args, output_prefix, lang, "idx", modality))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                nseq,
                ntok,
                100 * nunk / ntok,
                vocab.unk_word,
            )
        )
        for modality, ds in dss.items():
            print(f'\t{modality}')

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_dptree_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format != "binary":
            raise NotImplementedError(f'output format {args.output_format} not impl')

        make_binary_nstack_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    def make_all(lang, vocab):
        if args.trainpref:
            print(f'!!!! Warning..... Not during en-fr target because already done!.....')
            # make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)

        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.eval_workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.eval_workers)

    def make_all_src(lang, vocab):
        if args.trainpref:
            # print(f'!!!! Warning..... Not during en-fr source because already done!.....')
            make_dptree_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)

        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dptree_dataset(vocab, validpref, outprefix, lang, num_workers=args.eval_workers)

        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dptree_dataset(vocab, testpref, outprefix, lang, num_workers=args.eval_workers)

    def make_all_tgt(lang, vocab):
        make_all(lang, vocab)

    # make_all_src(args.source_lang, src_dict)
    print(f'|||| WARNIONG no processing for source.')
    if target:
        make_all_tgt(args.target_lang, tgt_dict)
        # print(f'No makign target')

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        raise NotImplementedError('alignfile Not impl at the moment')


def convert_raw(prefix, src_lang, tgt_lang, bpe_tree=False, bpe_code=None, unify_tree=False, workers=1,
                ignore_if_exist=True):
    raise NotImplementedError


"""
code  test.de  test.en  tmp  train.de  train.en  valid.de  valid.en
TEXT=examples/translation/iwslt14.tokenized.de-en
python preprocess.py --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en


# new here
ROOT_DIR=/data/nxphi47/projects/tree
TEXT=/data/nxphi47/projects/tree/raw_fairseq_tree2seq_ende
python preprocess_dptree2seq.py \
    --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --share_dict_txt $TEXT/share_vocab \
    --destdir ${ROOT_DIR}/data_fairseq
    --
    --srcdict ${TEXT}/share_vocab \
    --tgtdict ${TEXT}/share_vocab \



# WMT16 - new version
fairseq_vocab_bpe_en_de		newstest2014.tok.bpe.32000.en	newstest2016.tok.bpe.32000.en	test_en_2014			
total_vocab_bpe_en_de		train.tok.clean.bpe.32000.en	vocab.en

newstest2014.tok.bpe.32000.de	newstest2016.tok.bpe.32000.de	reduce_vocab_bpe_en_de		test_en_2016			
train.tok.clean.bpe.32000.de	train_en

fairseq_vocab_bpe_en_de        newstest2014.tok.bpe.32000.en  reduce_vocab_bpe_en_de  total_vocab_bpe_en_de         
train.tok.clean.bpe.32000.en        train.tok.clean.bpe.32000.testing.de
fairseq_vocab_bpe_en_de.wfreq  newstest2016.tok.bpe.32000.de  test_en_2014            train_en                      
train.tok.clean.bpe.32000.test2.de  train.tok.clean.bpe.32000.testing.en
newstest2014.tok.bpe.32000.de  newstest2016.tok.bpe.32000.en  test_en_2016            train.tok.clean.bpe.32000.de  
train.tok.clean.bpe.32000.test2.en  vocab.en

export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt16_ende_dptree
export BPE=${RAW_DIR}/bpe.32000
export train_r=${RAW_DIR}/train.bpe.sep.after-bpe
export valid_r=${RAW_DIR}/valid.bpe.sep.after-bpe
export test_r=${RAW_DIR}/test.bpe.sep.after-bpe





# for IWSLT-14

code  test.de  test.en  tmp  train.de  train.en  valid.de  valid.en

## this is for newly generated
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.bpe.sep.after-bpe
export valid_r=${RAW_DIR}/valid.bpe.sep.after-bpe
export test_r=${RAW_DIR}/test.bpe.sep.after-bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_translate_ende_iwslt14_bpe_v2

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_sep \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--bpe_code ${BPE} \
--no_remove_root \
--no_collapse \
--joined-dictionary \


--nwordssrc 32768 --nwordstgt 32768 \
--workers 32 \




--convert_raw \
--convert_raw_only \
--convert_with_bpe \

> prep_dptree_sep_iwslt14.txt 2>&1 &
--nwordssrc 32768 --nwordstgt 32768 \



# For En-Vi IWSLT-14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_envi
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.lower.tree
export valid_r=${RAW_DIR}/valid.lower.tree
export test_r=${RAW_DIR}/test.lower.tree

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_translate_envi_iwslt_32k

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_sep \
--source-lang en --target-lang vi \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \

--bpe_code ${BPE} \

# ================================ PREPPROCESS NSTACK2SEQ MERGE =============================== 

# For En-Fr IWSLT-15
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.tree.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.tree.bpe
export test_r=${RAW_DIR}/test.bpe32768.tree.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_nocolap

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang fr \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \

--srcdict ${OUT}/share_vocab \



# For Fr-En IWSLT-15
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.combined.tree-fren.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.tree-fren.bpe
export test_r=${RAW_DIR}/test.bpe32768.tree-fren.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_fren_iwslt_bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_fren_iwslt_bpe32k_nocolap

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang fr --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \




# !!! For En-De IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.tree.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.tree.bpe
export test_r=${RAW_DIR}/test.bpe32768.tree.bpe
# -----
export train_r=${RAW_DIR}/train.bpe.sep.after-bpe
export valid_r=${RAW_DIR}/valid.bpe.sep.after-bpe
export test_r=${RAW_DIR}/test.bpe.sep.after-bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_rmn
# ------ 
# export train_r=${RAW_DIR}/train.nobpe.p135.flat.bpetree
# export valid_r=${RAW_DIR}/valid.nobpe.p135.flat.bpetree
# export test_r=${RAW_DIR}/test.nobpe.p135.flat.bpetree
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_p135
# ------ 
# export train_r=${RAW_DIR}/train.nobpe.p330.flat.bpetree
# export valid_r=${RAW_DIR}/valid.nobpe.p330.flat.bpetree
# export test_r=${RAW_DIR}/test.nobpe.p330.flat.bpetree
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_p330
# ------ 
export train_r=${RAW_DIR}/train.bpe.sep.after-bpe
export valid_r=${RAW_DIR}/valid.bpe.sep.after-bpe
export test_r=${RAW_DIR}/test.bpe.sep.after-bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_nocolap


rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \
--no_remove_root \

--workers 8 \
--eval_workers 0 \




# || DE - EN IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-deen
export valid_r=${RAW_DIR}/valid.tree-deen
export test_r=${RAW_DIR}/test.tree-deen
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_deen_iwslt_32k_nocolap
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang de --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--no_remove_root \
--bpe_code ${BPE} \
--no_collapse \
--no_remove_root \


--workers 48 \
--eval_workers 0 \

# --no_collapse

#  --------------------------------------------------------------------------------




# |||||| En - Vi IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_envi
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.bpe32k.tree-afterbpe
export valid_r=${RAW_DIR}/tst2012.bpe32k.tree-afterbpe
export test_r=${RAW_DIR}/tst2013.bpe32k.tree-afterbpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_envi_iwslt_32k_sepvocab
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang vi \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--nwordssrc 32768 --nwordstgt 32768 \
--no_remove_root \
--bpe_code ${BPE} \
--workers 2 \
--eval_workers 0 \

--joined-dictionary \

# |||||| Zh - EN IWSLT'14
code                      parse.train.tree-encs.log  test.en      test.seg.zh        test.tree-zhen.zh            test.zh   train.seg.en  train.tree-zhen.en  train.tree-zhen.zh.raw.vocab  valid.en      valid.seg.zh        valid.tree-zhen.zh            valid.zh
parse.test.tree-encs.log  parse.valid.tree-encs.log  test.seg.en  test.tree-zhen.en  test.tree-zhen.zh.raw.vocab  train.en  train.seg.zh  train.tree-zhen.zh  train.zh                      valid.seg.en  valid.tree-zhen.en  valid.tree-zhen.zh.raw.vocab
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt15_enzh
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-zhen
export valid_r=${RAW_DIR}/valid.tree-zhen
export test_r=${RAW_DIR}/test.tree-zhen
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_zhen_iwslt_bpe32k
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang zh --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--nwordssrc 32768 --nwordstgt 32768 \
--no_remove_root \
--bpe_code ${BPE} \
--workers 2 \
--eval_workers 0 \



# |||||| En-Cs IWSLT'15
code                       test.cs                   test.tree-encs.bpe32k.en.before-bpe            train.en                              train.tree-encs.bpe32k.en.before-bpe.bpe.vocab  valid.tree-encs.bpe32k.cs                       valid.tree-encs.bpe32k.en.raw.vocab
parse.test.tree-encs.log   test.en                   test.tree-encs.bpe32k.en.before-bpe.bpe.vocab  train.tree-encs.bpe32k.cs             train.tree-encs.bpe32k.en.raw.vocab             valid.tree-encs.bpe32k.en
parse.train.tree-encs.log  test.tree-encs.bpe32k.cs  test.tree-encs.bpe32k.en.raw.vocab             train.tree-encs.bpe32k.en             valid.cs                                        valid.tree-encs.bpe32k.en.before-bpe
parse.valid.tree-encs.log  test.tree-encs.bpe32k.en  train.cs                                       train.tree-encs.bpe32k.en.before-bpe  valid.en                                        valid.tree-encs.bpe32k.en.before-bpe.bpe.vocab

export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_encs_proc
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-encs.bpe32k
export valid_r=${RAW_DIR}/test.tree-encs.bpe32k
export test_r=${RAW_DIR}/valid.tree-encs.bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_encs_iwslt_bpe32k
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang cs \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--nwordssrc 32768 --nwordstgt 32768 \
--no_remove_root \
--bpe_code ${BPE} \
--workers 2 \
--eval_workers 0 \
--joined-dictionary \






# --------------------------------------------------------------------------------------------------------------------


export ROOT_DIR=`pwd`
export TEXT=${ROOT_DIR}/t2t_data/translate_ende_wmt_bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/translate_ende_wmt16_bpe32k
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.tok.clean.bpe.32000 \
  --validpref $TEXT/newstest2013.tok.bpe.32000 \
  --testpref $TEXT/newstest2014.tok.bpe.32000 \
  --destdir $OUT \
  --srcdict \
  --joined-dictionary


  --nwordssrc 32768 --nwordstgt 32768 \
  
  
# WMT16-BPE EN-DE new
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt16_ende
export BPE=${RAW_DIR}/bpe.32000
export train_r=${RAW_DIR}/train.tok.clean.bpe.32000.combined.tree.bpetree
export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.tree.bpetree
export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.tree.bpetree

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe33k_v2
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_benepar_nocolap
export log_file=preprocess.wmt16ende.log
# rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 37000 --nwordstgt 37000 \
--bpe_code ${BPE} \
--workers 48 \
--eval_workers 0 \
--srcdict $OUT/dict.en.txt \

--tgtdict $OUT/dict.de.txt \


--workers 32 \
dict.de.txt  dict.en.txt



  
# WMT16-BPE EN-DE BenePar
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt16_ende
export BPE=${RAW_DIR}/bpe.32000
# export train_r=${RAW_DIR}/train.tok.clean.bpe.32000.benepar_en2_large
# export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.benepar_en2_large
# export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.benepar_en2_large
export train_r=${RAW_DIR}/train.tok.clean.bpe.32000.benepar_en2_large.v2.ende.bpe.w
export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.benepar_en2_large.v2.ende.bpe
export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.benepar_en2_large.v2.ende.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe33k_v2
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_benepar_nocolap
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_benepar
export log_file=preprocess.wmt16ende.log
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 37000 --nwordstgt 37000 \
--bpe_code ${BPE} \
--workers 32 \
--eval_workers 0 \
--no_remove_root \

--no_collapse \

# FIXME: Split point!
export percent=25
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt16_ende
export BPE=${RAW_DIR}/bpe.32000
export train_r=${RAW_DIR}/train.tree.p$percent
export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.tree.bpetree
export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.tree.bpetree
export tokens=32768
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe33k_v2
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_p$percent
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_fv3_p$percent
export log_file=preprocess.wmt16ende.log
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc $tokens --nwordstgt $tokens \
--bpe_code ${BPE} \
--workers 8 \
--eval_workers 0 \
--no_remove_root \




# ------------------------------------- IWSLT ------------------------------
# !!! For En-De IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.benepar_en2_large.ende.bpe
export valid_r=${RAW_DIR}/valid.benepar_en2_large.ende.bpe
export test_r=${RAW_DIR}/test.benepar_en2_large.ende.bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_benepar_nocolap


rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \
--no_remove_root \

--workers 8 \
--eval_workers 0 \


# ---- De-En IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/iwslt14.tokenized.de-en.v2
export BPE=${RAW_DIR}/code

export train_r=${RAW_DIR}/train.benepar_de.deen.bpe
export valid_r=${RAW_DIR}/valid.benepar_de.deen.bpe
export test_r=${RAW_DIR}/test.benepar_de.deen.bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_deen_iwslt_32k_benepar_nocolap

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang de --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \
--no_remove_root \


# --------
train.tok.clean.lower.bpe32768.benepar_en2_large.enfr.fr
test.bpe32768.benepar_en2_large.enfr.fr
valid.bpe32768.benepar_en2_large.enfr.fr

# !!! For En-Fr IWSLT'14
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.benepar_en2_large.enfr.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.benepar_en2_large.enfr.bpe
export test_r=${RAW_DIR}/test.bpe32768.benepar_en2_large.enfr.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_nocolap
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_benepar_nocolap

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang fr \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \


#  FOr Fr-En
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.benepar_fr.fren.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.benepar_fr.fren.bpe
export test_r=${RAW_DIR}/test.bpe32768.benepar_fr.fren.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_fren_iwslt_bpe32k
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_fren_iwslt_bpe32k_benepar_nocolap

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang fr --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_collapse \


# ====================== WMT'14 En-Fr ==============
train.prep250.prep.all.benepar_en2_large.enfr.bpe.w.en

export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt14_enfr_v3
export BPE=${RAW_DIR}/code
# export train_r=${RAW_DIR}/train.tok.clean.bpe.32000.benepar_en2_large
# export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.benepar_en2_large
# export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.benepar_en2_large
export train_r=${RAW_DIR}/train.prep250.prep.all.benepar_en2_large.enfr.bpe.w
export valid_r=${RAW_DIR}/valid.benepar_en2_large.enfr.bpe.w
export test_r=${RAW_DIR}/test.benepar_en2_large.enfr.bpe.w

# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe42k_benepar
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_wmt14_bpe42k_benepar

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_wmt14_bpeall_benepar
# rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang fr \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 0 --nwordstgt 0 \
--bpe_code ${BPE} \
--workers 32 \
--eval_workers 0 \
--no_remove_root \

--nwordssrc 42000 --nwordstgt 42000 \
--srcdict $OUT/dict.en.txt

--no_collapse \


# FIXING TEST SET
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt14_enfr_v3
export BPE=${RAW_DIR}/code
# export valid_r=${RAW_DIR}/valid.fixed
export valid_r=${RAW_DIR}/valid.fixed
export test_r=${RAW_DIR}/test.fixed
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_wmt14_bpe42k_benepar_infer
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_wmt14_bpeall_benepar_infer

# --trainpref ${train_r} \

python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang fr \
--user-dir ${user_dir} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--nwordssrc 0 --nwordstgt 0 \
--bpe_code ${BPE} \
--workers 0 \
--eval_workers 0 \
--no_remove_root \
--srcdict $OUT/dict.en.txt \
--tgtdict $OUT/dict.fr.txt

--joined-dictionary \
========================================================================================================================
new
#   IWSLT - En-De
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_ende_tree
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.bpe.sep.after-bpe
export valid_r=${RAW_DIR}/valid.bpe.sep.after-bpe
export test_r=${RAW_DIR}/test.bpe.sep.after-bpe
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_iwslt_32k_v3

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_remove_root \
--workers 8 \
--eval_workers 0 \

--no_collapse \

#   IWSLT - De-En
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_ende_tree
export BPE=${RAW_DIR}/code
export train_r=${RAW_DIR}/train.tree-deen
export valid_r=${RAW_DIR}/valid.tree-deen
export test_r=${RAW_DIR}/test.tree-deen
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_deen_iwslt_32k_v3

rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang de --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--no_remove_root \
--workers 8 \
--eval_workers 0 \

--no_collapse \

#  IWSLT - En-Fr
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr_tree
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.tree.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.tree.bpe
export test_r=${RAW_DIR}/test.bpe32768.tree.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_v3
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_nocolap
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_benepar_nocolap
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang fr \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \

#  IWSLT - Fr-En
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/translate_iwslt_enfr_tree
export BPE=${RAW_DIR}/bpe.32768
export train_r=${RAW_DIR}/train.tok.clean.lower.bpe32768.combined.tree-fren.bpe
export valid_r=${RAW_DIR}/valid.bpe32768.tree-fren.bpe
export test_r=${RAW_DIR}/test.bpe32768.tree-fren.bpe

export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_fren_iwslt_bpe32k_v3
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_nocolap
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_enfr_iwslt_bpe32k_benepar_nocolap
rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang fr --target-lang en \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \


# WMT - En - De
export ROOT_DIR=`pwd`
export PROJDIR=fi_fairseq
export user_dir=${ROOT_DIR}/${PROJDIR}
export RAW_DIR=${ROOT_DIR}/raw_data/wmt16_ende_v2
export BPE=${RAW_DIR}/bpe.32000
export train_r=${RAW_DIR}/train.tok.clean.bpe.32000.combined.tree.bpetree
export valid_r=${RAW_DIR}/newstest2013.tok.bpe.32000.tree.bpetree
export test_r=${RAW_DIR}/newstest2014.tok.bpe.32000.tree.bpetree

# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe33k_v2
# export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_benepar_nocolap
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe37k_fv3
export OUT=${ROOT_DIR}/data_fairseq_v2/nstack_merge_translate_ende_wmt16_bpe32k_fv3
export log_file=preprocess.wmt16ende.log
# rm -rf $OUT
python -m fi_fairseq.preprocess_nstack2seq_merge \
--source-lang en --target-lang de \
--user-dir ${user_dir} \
--trainpref ${train_r} \
--validpref ${valid_r} \
--testpref ${test_r} \
--destdir $OUT \
--no_remove_root \
--joined-dictionary \
--nwordssrc 32768 --nwordstgt 32768 \
--bpe_code ${BPE} \
--workers 16 \
--eval_workers 0 \

"""

# FIXME: get num updates:
#  ['optimizer_history'][-1]['num_updates']
"""
file = "train_fi_fairseq/nstack_merge_translate_enfr_iwslt_bpe32k/dwnstack_merge2seq_node_iwslt_onvalue_base_upmean_mean_mlesubenc_allcross_hierfull-transformer_base-b512-gpu1-upfre16-0fp16-id50msp512default/checkpoint_last.pt"
print(torch.load(file)['optimizer_history'][-1]['num_updates'])
"""


# if __name__ == "__main__":
#     parser = get_parser()
#     args = parser.parse_args()
#     main(args)

def truncate(x, y, s, e):
    with open(x, 'r') as fx:
        sents = fx.read().strip().split("\n")[s:e]
        with open(y, 'w') as fy:
            for s in sents:
                fy.write(f'{s}\n')


# # def try_parst
# import torch.nn as nn
# class SeparatePretrainedEmbedding(nn.Module):
#
#     def __init__(self, args, num_embeddings, embedding_dim, padding_idx, dictionary,
#                  pretrain_path, freeze=True, enforce_unk=False, same_dist=False):
#         super().__init__()
#         self.args = args
#         self.pretrain_dim = getattr(args, 'pretrain_dim', 300)
#         self.tune_epoch = getattr(args, 'tune_epoch', 1000000)
#         self.current_epoch = 0
#         self.finetuning = False
#         self.flip_switch = True
#
#         self.embedding_dim = embedding_dim
#         self.padding_idx = padding_idx
#         self.num_embeddings = num_embeddings
#         self.dictionary = dictionary
#         self.pretrain_path = pretrain_path
#         self.freeze = freeze
#         self.enforce_unk = enforce_unk
#         self.same_dist = same_dist
#
#         self.reordering, self.pretrained_emb, self.new_embed = self.build_embed(
#             num_embeddings, embedding_dim, padding_idx, dictionary, pretrain_path, self.tune_epoch
#         )
#
#         if self.pretrained_pad_idx is not None:
#             self.final_pad_idx = self.pretrained_pad_idx
#         else:
#             self.final_pad_idx = self.pretrained_emb.weight.size(0) + self.new_pad_idx
#
#         if self.pretrained_unk_idx is not None:
#             self.final_unk_idx = self.pretrained_unk_idx
#         else:
#             self.final_unk_idx = self.pretrained_emb.weight.size(0) + self.new_unk_idx
#
#     def create_new_embed(self, num_embeds, embedding_dim, pad_idx, pretrained_emb=None):
#         if self.same_


def cli_main():
    parser = options.get_preprocessing_parser()
    group = parser.add_argument_group('Preprocessing')

    group.add_argument("--convert_raw", action="store_true", help="convert_raw")
    group.add_argument("--convert_raw_only", action="store_true", help="convert_raw")
    group.add_argument("--convert_with_bpe", action="store_true", help="convert_with_bpe")
    # group.add_argument("--bpe_code", action="store_true", help="convert_with_bpe")
    group.add_argument('--bpe_code', metavar='FILE', help='bpe_code')

    group.add_argument("--no_remove_root", action="store_true", help="no_remove_root")
    group.add_argument("--no_take_pos_tag", action="store_true", help="no_take_pos_tag")
    group.add_argument("--no_take_nodes", action="store_true", help="no_take_nodes")
    group.add_argument("--no_reverse_node", action="store_true", help="no_reverse_node")
    group.add_argument("--no_collapse", action="store_true", help="no_collapse")

    group.add_argument("--raw_workers", metavar="N", default=0, type=int, help="number of parallel workers")
    group.add_argument("--eval_workers", metavar="N", default=0, type=int, help="number of parallel workers")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
