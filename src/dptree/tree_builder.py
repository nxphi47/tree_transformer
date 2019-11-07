from nltk import Tree

import torch

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk.tree import Tree
from nltk import tree, treetransforms
from copy import deepcopy
import queue

# from bllipparser import RerankingParser
import datetime
# rrp = RerankingParser.from_unified_model_dir("/home/ttnguyen/Downloads/WSJ-PTB3/")
from nltk.parse import CoreNLPParser
from nltk.parse import CoreNLPDependencyParser
import argparse
from shutil import copyfile

RETRIEVE_BATCH = int(os.environ.get('RETRIEVE_BATCH', 1000))
PARSER_TIMEOUT = int(os.environ.get('PARSER_TIMEOUT', 60000000))
PARSER_PORT = str(os.environ.get('PARSER_PORT', 9001))
# from .tree_process import *

SPECIAL_CHAR = {'&apos;': "'", '&apos;s': "'s", '&quot;': '"', '&#91;': '[',
                '&#93;': "]", '&apos;@@': "'@@", '&apos;t': "'t",
                '&amp;': "&", '&apos;ll': "'ll", '&apos;ve': "'ve",
                '&apos;m': "'m", '&apos;re': "'re", '&apos;d': "'d",
                '&#124;': "|", '&gt;': ">", '&lt;': "<"}

# special_character_dict['&quot;']="''"
SPECIAL_CHAR_MBACK = {v: k for k, v in SPECIAL_CHAR.items()}
SPECIAL_CHAR_MBACK['-LSB-'] = '&#91;'
SPECIAL_CHAR_MBACK['-RSB-'] = '&#93;'
SPECIAL_CHAR_MBACK['-LRB-'] = "("
SPECIAL_CHAR_MBACK['-RRB-'] = ")"
SPECIAL_CHAR_MBACK["''"] = "&quot;"


CORENLP_PARSER = None


class CusCoreNLPParser(CoreNLPParser):
    def api_call(self, data, properties=None, timeout=18000000):
        if properties is None:
            properties = {'parse.binaryTrees': "true"}
        return super().api_call(data, properties, timeout)
    @classmethod
    def build_parser(cls, port=9001):
        port = str(port)
        return cls(url=f'http://localhost:{port}')


def get_corenlp_parser():
    global CORENLP_PARSER
    if CORENLP_PARSER is None:
        print(f'!!!! Retrieving CoreNLPParser [{PARSER_PORT}], please make sure the server is running')
        CORENLP_PARSER = CusCoreNLPParser(url=f'http://localhost:{PARSER_PORT}')
    return CORENLP_PARSER


def replace_special_character(string):
    new_string = deepcopy(string)
    list_string = new_string.split(" ")
    new_list = deepcopy(list_string)
    for i in range(len(list_string)):
        for k, v in SPECIAL_CHAR.items():
            if k in list_string[i]:
                new_list[i] = list_string[i].replace(k, v)
    return " ".join(new_list)


def merge_list_tree(list_tree):
    root_label = [x.label() for x in list_tree]
    assert len(set(root_label)) == 1 and 'ROOT' in root_label
    list_string = "(ROOT " + " ".join([str(i) for i in range(len(list_tree))]) + ")"
    new_tree = Tree.fromstring(list_string)
    for i in range(len(list_tree)):
        new_tree[i] = list_tree[i][0]
    return new_tree


# def generate_tree_string(bpe_string, unify_tree=True):
#     word_string = bpe_string.replace("@@ ", "")
#     word_string = replace_special_character(word_string)
#     tree_string = list(get_corenlp_parser().parse_text(word_string, timeout=PARSER_TIMEOUT))
#     t1 = merge_list_tree(tree_string)
#     # t1.pretty_print()
#     tree_1 = deepcopy(t1)
#     for i in range(len(t1.leaves())):
#         if t1.leaves()[i] in SPECIAL_CHAR_MBACK:
#             t1[t1.leaf_treeposition(i)] = SPECIAL_CHAR_MBACK[t1.leaves()[i]]
#     tree_3 = deepcopy(t1)
#     parse_string = ' '.join(str(t1).split())
#     token_set = set(tree_3.leaves())
#     return parse_string, [tree_1, tree_3], token_set


def remap_chars(tree):
    for i in range(len(tree.leaves())):
        if tree.leaves()[i] in SPECIAL_CHAR_MBACK:
            tree[tree.leaf_treeposition(i)] = SPECIAL_CHAR_MBACK[tree.leaves()[i]]


def generate_tree_string_v2(bpe_string, unify_tree=True):
    word_string = bpe_string.replace("@@ ", "")
    word_string = replace_special_character(word_string)
    # tree_strings = list(get_corenlp_parser().parse_text(word_string, timeout=PARSER_TIMEOUT))
    tree_strings = list(get_corenlp_parser().parse_text(word_string))
    if unify_tree:
        merged = merge_list_tree(tree_strings)
        out_merged_tree = deepcopy(merged)
        remap_chars(merged)
        out_tree = deepcopy(merged)
        parse_string = ' '.join(str(merged).split())
        token_set = set(out_tree.leaves())
        return parse_string, [out_merged_tree, out_tree], token_set
    else:
        token_set = set()
        parse_strings = []
        befores = []
        afters = []
        for tree_s in tree_strings:
            before = deepcopy(tree_s)
            remap_chars(tree_s)
            after = deepcopy(tree_s)
            parse_string = ' '.join(str(tree_s).split())
            token_set = token_set.union(set(after.leaves()))
            parse_strings.append(parse_string)
            befores.append(before)
            afters.append(after)
        return parse_strings, [befores, afters], token_set


def generate_tree_string_v3(parser, bpe_string, unify_tree=True):
    word_string = bpe_string.replace("@@ ", "")
    word_string = replace_special_character(word_string)
    # tree_strings = list(get_corenlp_parser().parse_text(word_string, timeout=PARSER_TIMEOUT))
    tree_strings = list(parser.parse_text(word_string))
    if unify_tree:
        merged = merge_list_tree(tree_strings)
        out_merged_tree = deepcopy(merged)
        remap_chars(merged)
        out_tree = deepcopy(merged)
        parse_string = ' '.join(str(merged).split())
        token_set = set(out_tree.leaves())
        return parse_string, [out_merged_tree, out_tree], token_set
    else:
        token_set = set()
        parse_strings = []
        befores = []
        afters = []
        for tree_s in tree_strings:
            before = deepcopy(tree_s)
            remap_chars(tree_s)
            after = deepcopy(tree_s)
            parse_string = ' '.join(str(tree_s).split())
            token_set = token_set.union(set(after.leaves()))
            parse_strings.append(parse_string)
            befores.append(before)
            afters.append(after)
        return parse_strings, [befores, afters], token_set


class DPTreeDataset(Dataset):
    def __init__(self, data_file, transform=None, unify_tree=True, port=PARSER_PORT):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = open(data_file, "r").readlines()
        print(f'Finish reading lines [port={port}]: {len(self.data)}')
        print(f'Line 0: {self.data[0]}')
        self.transform = transform
        self.port = port
        self.unify_tree = unify_tree
        self.parser = get_corenlp_parser()
        self.parser = CusCoreNLPParser(url=f'http://localhost:{self.port}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].rstrip("\n")
        # if self.transform:
        try:
            parse_string, _, token_string_set = generate_tree_string_v3(self.parser, sample, unify_tree=self.unify_tree)
        except Exception as e:
            print(f'Error happen at index {idx}')
            print(f'{sample}')
            raise e

        sample = {
            'pstring': parse_string,
            'token_set': list(token_string_set)
        }
        return sample


def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    return tree_string


def tree_from_string(s):
    try:
        tree_string = s
        tree_string = tree_str_post_process(tree_string)
        tree_line = Tree.fromstring(tree_string)
    except Exception as e:
        # print(f'Tree.fromstring(tree_string) failed, try to omit the post_process')
        # print(s)
        tree_string = s
        tree_line = Tree.fromstring(tree_string)
    return tree_line


class TreeBuilder(object):
    # SENT_SPLITTER = '######'
    SENT_SPLITTER = '#####------#####'

    def __init__(
            self, transform=None, unify_tree=True, bpe_tree=False, bpe_code=None) -> None:
        super().__init__()
        self.transform = transform
        self.bpe_tree = bpe_tree
        self.bpe_code = bpe_code
        self.unify_tree = unify_tree

        if self.bpe_tree:
            assert bpe_code is not None and os.path.exists(bpe_code)

    def retrieve_tree_data(self, dataloader):
        data = []
        vocab = set()

        for i_batch, sample_batched in enumerate(dataloader):
            if self.unify_tree:
                assert isinstance(sample_batched['pstring'], (list, tuple))
                s = sample_batched['pstring'][0]
                assert isinstance(s, str)
            else:
                try:
                    s = [x[0] for x in sample_batched['pstring']]
                    assert isinstance(s[0], str), f'{sample_batched["pstring"]}'
                except Exception as e:
                    print(f'Failed step {i_batch} -> continue')
                    print(sample_batched)
                    # raise e
                    continue

            v = sample_batched['token_set'][0]

            if i_batch % RETRIEVE_BATCH == 0:
                print(f'Retrieve batch [{i_batch}]: time {datetime.datetime.now()}]')

            data.append(s)
            # vocab = vocab.union(set(sample_batched[1]))
            vocab = vocab.union(set(v))
            #
            # raise ValueError

        vocab = list(vocab)
        return data, vocab

    def build_bpe_tree_vocab(self, raw_vocab_file, output_file):
        # raw_vocab_file = f'{output_file}.raw.vocab'
        bpe_vocab_file = f'{output_file}.bpe.vocab'

        print(f'Applying BPE: subword-nmt apply-bpe -c {self.bpe_code} < {raw_vocab_file} > {bpe_vocab_file}')
        os.system(f'subword-nmt apply-bpe -c {self.bpe_code} < {raw_vocab_file} > {bpe_vocab_file}')
        assert os.path.exists(bpe_vocab_file)

        # re-open bpe vocab
        with open(bpe_vocab_file, "r") as f:
            vocab_bpe = f.readlines()
        list_dict = [x.strip().replace("@@ ", "") for x in vocab_bpe]
        word2bpe = {}

        for i, w in enumerate(list_dict):
            word2bpe[w] = vocab_bpe[i].strip().split(" ")

        return bpe_vocab_file, word2bpe, list_dict, vocab_bpe

    def apply_bpe_on_tree_strings(self, data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe, multiple_sents=False):
        parse_string_data = []

        def parse(x, index=0):
            try:
                tree_i = tree_from_string(x)
            except Exception as e:
                print(f'Error when parse tree at index {index}')
                print(f'{x}')
                raise e
            new_tree_i = deepcopy(tree_i)
            for j in range(len(tree_i.leaves())):
                if tree_i.leaves()[j] in word2bpe and len(word2bpe[tree_i.leaves()[j]]) > 1:
                    loc_leaf_j = tree_i.leaf_treeposition(j)
                    label_leaf_j = tree_i[loc_leaf_j[:-1]].label()
                    bpe_tree_j = Tree(label_leaf_j,
                                      [Tree(label_leaf_j + '_bpe', [x]) for x in word2bpe[tree_i.leaves()[j]]])
                    new_tree_i[loc_leaf_j[:-1]] = bpe_tree_j
            parsing_tree = ' '.join(str(new_tree_i).split())
            return parsing_tree

        for i, d in enumerate(data):
            if i % 100000 == 0:
                print(i)

            if isinstance(d, list):
                parse_string_data.append([parse(x, index=i) for x in d])
            else:
                parse_string_data.append(parse(d, index=i))

        return parse_string_data

    def export_seq_file(self, data, file):
        with open(file, "w") as f:
            for i, w in enumerate(data):
                f.write(f'{w}\n')
            f.close()

    def export_separate_seq_file(self, data, file):
        with open(file, "w") as f:
            for i, w in enumerate(data):
                assert isinstance(w, (list, tuple))
                s = self.__class__.SENT_SPLITTER.join(w)
                f.write(f'{s}\n')
            f.close()

    def read_separate_file(self, file):
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data = [x.split(self.__class__.SENT_SPLITTER) for x in lines]

        return data

    def export_text_to_unified_tree_string(self, input_file, output_file, num_workers=1):
        dptree_dataset = DPTreeDataset(input_file, self.transform, unify_tree=True)
        dataloader = DataLoader(dptree_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        data, vocab = self.retrieve_tree_data(dataloader)

        raw_vocab_file = f'{output_file}.raw.vocab'
        print(f'Generating raw vocab: {raw_vocab_file}')
        self.export_seq_file(vocab, raw_vocab_file)
        print(f'Generate tree string data: {output_file}')
        self.export_seq_file(data, output_file)

        if self.bpe_tree:
            bpe_output_file = f'{output_file}.bpe'
            print(f'Proceed to generate bpe tree: {bpe_output_file}')
            bpe_vocab_file, word2bpe, list_dict, vocab_bpe = self.build_bpe_tree_vocab(raw_vocab_file, output_file)

            print(f'apply_bpe_on_tree_strings: {bpe_output_file}')
            parse_string_data = self.apply_bpe_on_tree_strings(
                data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe
            )
            self.export_seq_file(parse_string_data, bpe_output_file)

    def build_separate_bpe_tree(self, data, raw_vocab_file, before_bpe_output_file, bpe_output_file):
        print(f'Proceed to generate bpe tree: {bpe_output_file}')

        bpe_vocab_file, word2bpe, list_dict, vocab_bpe = self.build_bpe_tree_vocab(
            raw_vocab_file, before_bpe_output_file)

        print(f'apply_bpe_on_tree_strings: {bpe_output_file}')
        parse_string_data = self.apply_bpe_on_tree_strings(
            data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe
        )
        self.export_separate_seq_file(parse_string_data, bpe_output_file)

    def export_text_to_separate_tree_strings(self, input_file, output_file, num_workers=1, ignore_if_exist=False, port=PARSER_PORT):
        raw_vocab_file = f'{output_file}.raw.vocab'
        before_bpe_output_file = f'{output_file}.before-bpe'

        if ignore_if_exist:
            if self.bpe_tree and os.path.exists(before_bpe_output_file):
                assert os.path.exists(raw_vocab_file)
                print(f'Ignore tree generation, proceed to tree production')
                bpe_output_file = output_file
                data = self.read_separate_file(before_bpe_output_file)
                self.build_separate_bpe_tree(data, raw_vocab_file, before_bpe_output_file, bpe_output_file)

        dptree_dataset = DPTreeDataset(input_file, self.transform, unify_tree=False, port=port)

        dataloader = DataLoader(dptree_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        data, vocab = self.retrieve_tree_data(dataloader)
        print(f'Generating raw vocab: {raw_vocab_file}')
        self.export_seq_file(vocab, raw_vocab_file)
        print(f'Generate tree string data: {output_file}')
        self.export_separate_seq_file(data, output_file)

        if self.bpe_tree:
            # before_bpe_output_file = f'{output_file}.before-bpe'
            copyfile(output_file, before_bpe_output_file)
            # bpe_output_file = f'{output_file}.bpe'
            bpe_output_file = output_file

            self.build_separate_bpe_tree(data, raw_vocab_file, before_bpe_output_file, bpe_output_file)

            # print(f'Proceed to generate bpe tree: {bpe_output_file}')
            # bpe_vocab_file, word2bpe, list_dict, vocab_bpe = self.build_bpe_tree_vocab(raw_vocab_file, before_bpe_output_file)
            # print(f'apply_bpe_on_tree_strings: {bpe_output_file}')
            # parse_string_data = self.apply_bpe_on_tree_strings(
            #     data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe
            # )
            # self.export_separate_seq_file(parse_string_data, bpe_output_file)


class CoreNLPTreeBuilder(TreeBuilder):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file')
    parser.add_argument('--out_file')
    parser.add_argument('--port', type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    args = parser.parse_args()

    print(f'Parser port: {args.port}')
    # builder = TreeBuilder(transform=True, bpe_tree=False, bpe_code=None)
    builder = TreeBuilder(transform=True, bpe_tree=False, bpe_code=False, unify_tree=False)
    builder.export_text_to_separate_tree_strings(
        args.in_file, args.out_file, num_workers=args.num_workers, port=args.port)

#
# if __name__ == '__main__':
#     print('testing')
#     input_file = os.environ['input_file']
#     output_file = os.environ['output_file']
#     bpe_code = os.environ['bpe_code']
#
    # builder = TreeBuilder(transform=True, bpe_tree=True, bpe_code=bpe_code)
#     # builder.export_text_to_unified_tree_string(input_file, output_file, num_workers=0)
#
#     eg1 = 'The quick brown fox jumps over the lazy dog .'
#     eg2 = 'The quick brown fox jumps over the lazy dog . Meanwhile , the cat is sitting on the mat .'
#     builder = TreeBuilder(transform=True, bpe_tree=True, bpe_code=bpe_code, unify_tree=False)
#     builder.export_text_to_separate_tree_strings(input_file, output_file, num_workers=0)
