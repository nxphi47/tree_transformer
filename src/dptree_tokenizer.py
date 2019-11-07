# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import os, re

import datetime
import torch
from multiprocessing import Pool
from nltk import Tree
from torch.utils import data
from fairseq.data import Dictionary

from .dptree import tree_process, tree_builder

SPACE_NORMALIZER = re.compile(r"\s+")

PRINT_INTERVAL = int(os.environ.get('PRINT_INTERVAL', 10000))

PLACE_HOLDER = '<placeholder>'


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    return tree_string


def try_parse(s):
    try:
        tree_string = s
        tree_string = tree_str_post_process(tree_string)
        tree_line = Tree.fromstring(tree_string)
    except Exception as e:
        print(f'Tree.fromstring(tree_string) failed, try to omit the post_process')
        print(s)
        try:
            tree_string = s
            tree_line = Tree.fromstring(tree_string)
        except Exception as e:
            print(f'ERROR: unable to parse the tree')
            print(s)
            raise e

    return tree_line


def read_parse(file):
    with open(file, 'r') as f:
        sents = f.read().strip().split('\n')
        lines = [try_parse(x) for x in sents]
    return lines


def tree_from_string(s):
    try:
        tree_string = s
        tree_string = tree_str_post_process(tree_string)
        tree_line = Tree.fromstring(tree_string)
    except Exception as e:
        # print(f'Tree.fromstring(tree_string) failed, try to omit the post_process')
        # print(s)
        # tree_string = s
        # tree_line = Tree.fromstring(tree_string)
        try:
            tree_string = s
            tree_line = Tree.fromstring(tree_string)
        except Exception as e:
            print(f'ERROR: unable to parse the tree')
            print(s)
            raise e
    return tree_line


class DPTreeTokenizer(object):

    @staticmethod
    def line2example(s, vocab, consumer, tokenize=tokenize_line,
                     append_eos=False, reverse_order=False,
                     add_if_not_exist=False,
                     offset=0, end=-1, noat=False, cnf=True):

        tree_string = s
        tree_line = tree_from_string(tree_string)

        tree_process.padding_leaves(tree_line)
        tree_line = tree_process.clean_node(tree_line)

        line_leaves, line_matrix, line_node_label, line_leaf_label = tree_process.tree2matrix(tree_line, cnf=cnf)

        # try:
        line_node, line_label, line_index = tree_process.generate_data(tree_line, cnf)
        # except RecursionError as e:

        if noat:
            line_node = [x[1:] if x[0] == '@' else x for x in line_node]

        node_indices = DPTreeTokenizer.tokenize(
            words=line_node,
            vocab=vocab,
            tokenize=tokenize,
            add_if_not_exist=add_if_not_exist,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        labels_indices = DPTreeTokenizer.tokenize(
            words=line_label,
            vocab=vocab,
            tokenize=tokenize,
            add_if_not_exist=add_if_not_exist,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        line_length = len(line_leaves)

        # TODO: add pads
        # FIXME: MUST CHECK pad_index = 1 systematically!
        pad_index = 1
        node_indices = torch.cat([node_indices, torch.tensor([pad_index]).int()], 0)
        labels_indices = torch.cat([labels_indices, torch.tensor([pad_index]).int()], 0)
        line_index += [[line_length, line_length]]

        line_indices = torch.tensor(line_index).int()
        line_len = torch.tensor([line_length]).int()

        example = {
            "nodes": node_indices,
            "labels": labels_indices,
            "indices": line_indices,
            "length": line_len
        }
        return example

    @staticmethod
    def convert2stt5_lines(line_node, line_label):
        line_tokens = []
        line_sentiments = []

        for i, (x, y) in enumerate(zip(line_node, line_label)):
            # pad in line_label is phrase_label in line_node
            if y == '<pad>':
                line_tokens.append(PLACE_HOLDER)
                line_sentiments.append(int(x.replace('_node_label', '')))
            else:
                line_tokens.append(x)
                line_sentiments.append(int(y.replace('_leaf_label', '')))

        return line_tokens, line_sentiments

    @staticmethod
    def line2example_sst5(s, vocab, consumer, tokenize=tokenize_line,
                          append_eos=False, reverse_order=False,
                          add_if_not_exist=False,
                          offset=0, end=-1, noat=False):
        """This one has no node-label, instead return objects that has labeling index"""
        tree_string = s
        # tree_string = tree_str_post_process(tree_string)
        tree_line = tree_from_string(tree_string)

        tree_process.padding_leaves(tree_line)
        tree_line = tree_process.clean_node(tree_line)
        line_leaves, line_matrix, line_node_label, line_leaf_label = tree_process.tree2matrix(tree_line)
        line_node, line_label, line_index = tree_process.generate_data(tree_line)

        line_tokens, line_sentiments = DPTreeTokenizer.convert2stt5_lines(line_node, line_label)

        if noat:
            line_tokens = [x[1:] if x[0] == '@' else x for x in line_tokens]

        node_indices = DPTreeTokenizer.tokenize(
            words=line_tokens,
            vocab=vocab,
            tokenize=tokenize,
            add_if_not_exist=add_if_not_exist,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        if append_eos:
            line_sentiments.append(0)

        sentiment_indices = torch.tensor(line_sentiments).int()

        line_length = len(line_leaves)

        # TODO: add pads
        # FIXME: MUST CHECK pad_index = 1 systematically!
        pad_index = 1
        node_indices = torch.cat([node_indices, torch.tensor([pad_index]).int()], 0)
        sentiment_indices = torch.cat([sentiment_indices, torch.tensor([pad_index]).int()], 0)
        line_index += [[line_length, line_length]]

        line_indices = torch.tensor(line_index).int()
        line_len = torch.tensor([line_length]).int()

        example = {
            "nodes": node_indices,
            "labels": sentiment_indices,
            "indices": line_indices,
            "length": line_len
        }
        return example

    @staticmethod
    def line2example_sst5_plb(
            s, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1):
        """This one has node-label, instead return objects that has labeling index"""
        tree_string = s
        # tree_string = tree_str_post_process(tree_string)
        tree_line = tree_from_string(tree_string)

        tree_process.padding_leaves(tree_line)
        tree_line = tree_process.clean_node(tree_line)
        line_leaves, line_matrix, line_node_label, line_leaf_label = tree_process.tree2matrix(tree_line)
        line_node, line_label, line_index = tree_process.generate_data(tree_line)

        line_tokens, line_sentiments = DPTreeTokenizer.convert2stt5_lines(line_node, line_label)

        node_indices = DPTreeTokenizer.tokenize(
            words=line_tokens,
            vocab=vocab,
            tokenize=tokenize,
            add_if_not_exist=add_if_not_exist,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )

        if append_eos:
            line_sentiments.append(0)

        sentiment_indices = torch.tensor(line_sentiments).int()

        line_length = len(line_leaves)

        # TODO: add pads
        # FIXME: MUST CHECK pad_index = 1 systematically!
        pad_index = 1
        node_indices = torch.cat([node_indices, torch.tensor([pad_index]).int()], 0)
        sentiment_indices = torch.cat([sentiment_indices, torch.tensor([pad_index]).int()], 0)
        line_index += [[line_length, line_length]]

        line_indices = torch.tensor(line_index).int()
        line_len = torch.tensor([line_length]).int()

        example = {
            "nodes": node_indices,
            "labels": sentiment_indices,
            "indices": line_indices,
            "length": line_len
        }
        return example

    @staticmethod
    def line2multi_example(
            s, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1,
            line2example=None,
            noat=False,
            cnf=True,
    ):

        tree_strings = s.split(tree_builder.TreeBuilder.SENT_SPLITTER)

        try:
            if line2example is None:
                line2example = DPTreeTokenizer.line2example
            examples = [line2example(
                x, vocab, consumer, tokenize, append_eos, reverse_order, add_if_not_exist,
                offset, end, noat=noat, cnf=cnf,
            ) for x in tree_strings]
        except ValueError as ve:
            print(f'Error in this example')
            print(s)
            raise ve

        keys = list(examples[0].keys())

        def merge_trees(tensors, pad_idx=1):
            """
                [(n1, d...), (n2, d...), (nm]
            :param tensors:
            :return: [m, max(n1...nm), d...]
            """
            size = max(v.size(0) for v in tensors)
            rest_size = list(tensors[0].size()[1:])
            res = tensors[0].new(len(tensors), size, *rest_size).fill_(pad_idx)

            def copy_tensor(src, dst):
                assert dst.numel() == src.numel()
                dst.copy_(src)

            for i, v in enumerate(tensors):
                copy_tensor(v, res[i][:len(v)])
            return res

        ntok = sum([len(x['nodes']) for x in examples])

        examples_d = {
            "nodes": merge_trees([x['nodes'] for x in examples], 1),
            "labels": merge_trees([x['labels'] for x in examples], 1),
            "indices": merge_trees([x['indices'] for x in examples], 0),
            "length": torch.cat([x['length'] for x in examples], 0),
        }
        return examples_d, ntok

    @staticmethod
    def line2leaves_n_nodes(s, noat=False, cnf=True):
        line_nodes = []
        line_labels = []

        strings = s.split(tree_builder.TreeBuilder.SENT_SPLITTER)
        for s in strings:

            tree_string = s

            # tree_string = tree_str_post_process(tree_string)
            tree_line = tree_from_string(tree_string)
            tree_process.padding_leaves(tree_line)
            tree_line = tree_process.clean_node(tree_line)

            try:
                line_node, line_label, line_index = tree_process.generate_data(tree_line, cnf=cnf)
                if noat:
                    line_node = [x[1:] if x[0] == '@' else x for x in line_node]
            except IndexError as e:
                print(tree_string)
                tree_line.pretty_print()
                raise e
            except RecursionError as er:
                print("Recursion error due to too long tree -> omit the tree")
                continue

            line_nodes += line_node
            line_labels += line_label

        return line_nodes, line_labels

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, vocab, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                vocab.add_symbol(w, c)

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    DPTreeTokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, vocab.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(DPTreeTokenizer.add_file_to_dictionary_single_worker(filename, tokenize, vocab.eos_word))

    @staticmethod
    def binarize(filename, vocab, consumer, tokenize=tokenize_line,
                 append_eos=False, reverse_order=False,
                 add_if_not_exist=False,
                 offset=0, end=-1, **kwargs):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                example = DPTreeTokenizer.line2example(
                    s=line,
                    vocab=vocab,
                    consumer=replaced_consumer,
                    tokenize=tokenize,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    add_if_not_exist=add_if_not_exist,
                    offset=offset,
                    end=end
                )
                nseq += 1
                ntok += len(example['nodes'])
                consumer(example)
                line = f.readline()

                if nseq == 1 or nseq % PRINT_INTERVAL == 0:
                    now = str(datetime.datetime.now().time())
                    print(f'Dptree:binarize:[{now}] offset={offset}: '
                          f'nseq={nseq}, dict={len(vocab)}')

        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_separate(
            filename, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1,
            worker_id=None
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if 0 < end < f.tell():
                    break

                example, ntok_ = DPTreeTokenizer.line2multi_example(
                    s=line,
                    vocab=vocab,
                    consumer=replaced_consumer,
                    tokenize=tokenize,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    add_if_not_exist=add_if_not_exist,
                    offset=offset,
                    end=end
                )

                nseq += 1
                # ntok += sum([len(x) for x in example['nodes']])
                ntok += ntok_
                consumer(example)
                line = f.readline()

                if nseq == 1 or nseq % PRINT_INTERVAL == 0:
                    now = str(datetime.datetime.now().time())
                    print(f'Dptree:binarize:[{now}]-[worker_id={worker_id}] offset={offset}: '
                          f'nseq={nseq}, dict={len(vocab)}')

        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def acquire_vocab(filename, consumer, tokenize=tokenize_line,
                      append_eos=False, reverse_order=False,
                      add_if_not_exist=True,
                      offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        vocab = Dictionary()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                line_node, line_label = DPTreeTokenizer.line2leaves_n_nodes(line)

                node_indices = DPTreeTokenizer.tokenize(
                    words=line_node,
                    vocab=vocab,
                    tokenize=tokenize,
                    add_if_not_exist=add_if_not_exist,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                labels_indices = DPTreeTokenizer.tokenize(
                    words=line_label,
                    vocab=vocab,
                    tokenize=tokenize,
                    add_if_not_exist=add_if_not_exist,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                nseq += 1
                # ntok += len(example['nodes'])
                # consumer(example)
                line = f.readline()

                if nseq == 1 or nseq % PRINT_INTERVAL == 0:
                    print(f'Dptree:binarize: offset={offset}: nseq={nseq}, dict={len(vocab)}')

        return vocab

    @staticmethod
    def acquire_vocab_stt5(
            filename, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=True,
            offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        vocab = Dictionary()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                line_node, line_label = DPTreeTokenizer.line2leaves_n_nodes(line)
                line_tokens, line_sentiments = DPTreeTokenizer.convert2stt5_lines(line_node, line_label)
                node_indices = DPTreeTokenizer.tokenize(
                    words=line_tokens,
                    vocab=vocab,
                    tokenize=tokenize,
                    add_if_not_exist=add_if_not_exist,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                line = f.readline()

                if nseq == 1 or nseq % PRINT_INTERVAL == 0:
                    now = str(datetime.datetime.now().time())
                    print(f'Dptree:binarize:[{now}] offset={offset}: '
                          f'nseq={nseq}, dict={len(vocab)}')

        return vocab

    @staticmethod
    def acquire_vocab_multithread(
            filename, vocab, tokenize=tokenize_line, num_workers=1,
            add_single_thread=None, noat=False, cnf=True,
    ):
        def merge_result(counter):
            for w, c in counter.items():
                vocab.add_symbol(w, c)

        if add_single_thread is None:
            add_single_thread = DPTreeTokenizer.add_to_vocab_single_thread

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    add_single_thread,
                    (filename, tokenize, vocab.eos_word, worker_id, num_workers, noat, cnf)
                ))
            pool.close()
            pool.join()
            assert len(results) == num_workers, f'{len(results)} processes died!'
            for r in results:
                merge_result(r.get())
        else:
            merge_result(DPTreeTokenizer.add_to_vocab_single_thread(filename, tokenize, vocab.eos_word, noat=noat, cnf=cnf))

    @staticmethod
    def add_to_vocab_single_thread(filename, tokenize, eos_word, worker_id=0, num_workers=1, noat=False, cnf=True):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                line_node, line_label = DPTreeTokenizer.line2leaves_n_nodes(line, noat=noat, cnf=cnf)
                for word in line_node:
                    counter.update([word])
                counter.update([eos_word])
                for word in line_label:
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_to_vocab_single_thread_stt5(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f)  # drop first incomplete line
            line = f.readline()
            while line:
                line_node, line_label = DPTreeTokenizer.line2leaves_n_nodes(line)
                line_tokens, line_sentiments = DPTreeTokenizer.convert2stt5_lines(line_node, line_label)

                for word in line_tokens:
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(words, vocab, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        # words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = vocab.add_symbol(word)
            else:
                idx = vocab.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = vocab.eos_index
        return ids

# class BinarizerDataset(data.Dataset):
#     def __init__(
#             self,
#             filename,
#             vocab, replaced_consumer,
#             tokenize=tokenize_line,
#             append_eos=False, reverse_order=False,
#             add_if_not_exist=False,
#             ) -> None:
#         super().__init__()
#         self.filename = filename
#         self.vocab = vocab
#         self.tokenize_line = tokenize
#         self.replaced_consumer = replaced_consumer
#         self.append_eos = append_eos
#         self.reverse_order = reverse_order
#         self.add_if_not_exist = add_if_not_exist
#
#         self.data_lines = self.retrieve_lines()
#
#     def retrieve_lines(self):
#         lines = []
#         with open(self.filename, 'r') as f:
#             line = safe_readline(f)
#             while line:
#                 lines.append(line)
#
#                 line = f.readline()
#         return lines
#
#     def __len__(self) -> int:
#         return len(self.data_lines)
#
#     @staticmethod
#     def tokenize(words, vocab, tokenize=tokenize_line, add_if_not_exist=True,
#                  consumer=None, append_eos=True, reverse_order=False):
#         if reverse_order:
#             words = list(reversed(words))
#         nwords = len(words)
#         ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
#
#         for i, word in enumerate(words):
#             if add_if_not_exist:
#                 idx = vocab.add_symbol(word)
#             else:
#                 idx = vocab.index(word)
#             if consumer is not None:
#                 consumer(word, idx)
#             ids[i] = idx
#         if append_eos:
#             ids[nwords] = vocab.eos_index
#         return ids
#
#     def __getitem__(self, index: int):
#         line = self.data_lines[index]
#         ids = self.vocab.encode_line(
#             line=line,
#             line_tokenizer=self.tokenize_line,
#             add_if_not_exist=False,
#             consumer=self.replaced_consumer,
#             append_eos=self.append_eos,
#             reverse_order=self.reverse_order,
#         )
#         return ids
#
#     @classmethod
#     def export_binarized_dataset(
#             cls, filename,
#             vocab, consumer,
#             tokenize=tokenize_line,
#             append_eos=False, reverse_order=False,
#             add_if_not_exist=False, num_workers=0
#     ):
#         nseq, ntok = 0, 0
#         stat = {'ntok': 0}
#         replaced = Counter()
#
#         def replaced_consumer(word, idx):
#             if idx == vocab.unk_index and word != vocab.unk_word:
#                 replaced.update([word])
#             stat['ntok'] += 1
#
#         dataset = cls(
#             filename, vocab, replaced_consumer,
#             tokenize, append_eos, reverse_order, add_if_not_exist
#         )
#
#         dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
#         for i, batch in enumerate(dataloader):
#             if i == 1 or i % PRINT_INTERVAL == 0:
#                 now = str(datetime.datetime.now().time())
#                 print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}]')
#
#             if isinstance(batch, dict):
#                 sample = {k: v[0] for k, v in batch.items()}
#             else:
#                 sample = batch[0]
#             nseq += 1
#             consumer(sample)
#
#         end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
#         return end_stat
