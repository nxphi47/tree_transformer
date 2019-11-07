from collections import Counter
import os, re

import datetime
import torch
from multiprocessing import Pool
from nltk import Tree
from torch.utils import data
from fairseq.data import Dictionary

from .dptree import tree_process, tree_builder
from .dptree_tokenizer import DPTreeTokenizer
from .nstack_tokenizer import NstackTreeTokenizer
import time

SPACE_NORMALIZER = re.compile(r"\s+")
PRINT_INTERVAL = int(os.environ.get('PRINT_INTERVAL', 10000))


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


class Stat(object):

    def __init__(self) -> None:
        super().__init__()
        self.ntok = 0


class BinarizerDataset(data.Dataset):
    def __init__(
            self,
            filename,
            vocab,
            replaced_consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
    ) -> None:
        super().__init__()
        self.filename = filename
        self.vocab = vocab
        self.tokenize_line = tokenize
        self.replaced_consumer = replaced_consumer
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.add_if_not_exist = add_if_not_exist

        self.data_lines = self.retrieve_lines()

    def _retrieve_lines_from_file(self, file):
        lines = []
        with open(file, 'r') as f:
            line = safe_readline(f)
            while line:
                lines.append(line)
                line = f.readline()
        return lines

    def retrieve_lines(self):
        return self._retrieve_lines_from_file(self.filename)

    def __len__(self) -> int:
        return len(self.data_lines)

    @staticmethod
    def tokenize(words, vocab, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
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

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        ids = self.vocab.encode_line(
            line=line,
            line_tokenizer=self.tokenize_line,
            add_if_not_exist=False,
            consumer=self.replaced_consumer,
            append_eos=self.append_eos,
            reverse_order=self.reverse_order,
        )
        ntok_ = len(ids)
        return ids, ntok_

    @classmethod
    def export_binarized_dataset(
            cls, filename,
            vocab,
            consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,

    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                eq = word == '-'
                print(f'Update: |{word}| eq-: {eq}')
                replaced.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}]')

            if isinstance(example, dict):
                sample = {k: v[0] for k, v in example.items()}
            else:
                sample = example[0]
            assert ntok_  > 0, f'ntok! {ntok_} [{i}]: {example}'
            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        return end_stat


class DPTreeSeparateBinarizerDataset(BinarizerDataset):

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            example, ntok_ = DPTreeTokenizer.line2multi_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            raise er

        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls, filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        stat = {'ntok': 0}
        # stat = Stat()
        replaced = Counter()
        ntok_counter = Counter()

        def replaced_consumer(word, idx):
            # global stat
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')

            sample = {k: v[0] for k, v in example.items()}
            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        # ntok = sum(ntok_counter.values())

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        return end_stat


class DPTreeSeparateBinarizerImdbDataset(DPTreeSeparateBinarizerDataset):
    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            example, ntok_ = DPTreeTokenizer.line2multi_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            example = {
                'nodes': torch.zeros(1, dtype=torch.int32),
                'labels': torch.zeros(1, dtype=torch.int32),
                'indices': torch.zeros(1, 2, dtype=torch.int32),
                'length': torch.zeros(1, dtype=torch.int32),
            }
            ntok_ = 0

        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls, filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        stat = {'ntok': 0}
        # stat = Stat()
        replaced = Counter()
        ntok_counter = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist
        )
        skipped = []

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')

            sample = {k: v[0] for k, v in example.items()}

            if sample['nodes'].sum() == 0:
                print(f'Skipping line {i}')
                skipped_line = dataset.data_lines[i]
                skipped.append(skipped_line)
                continue

            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        # ntok = sum(ntok_counter.values())

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)

        skipped_file = f'{filename}.processed.skipped'
        print(f'Writing skipped lines into {skipped_file}')
        with open(skipped_file, 'w') as f:
            f.write('\n'.join(skipped))

        return end_stat


class DPTreeSeparateSST5BinarizerDataset(DPTreeSeparateBinarizerDataset):

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            example, ntok_ = DPTreeTokenizer.line2multi_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                line2example=DPTreeTokenizer.line2example_sst5
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve

        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls,
            filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            nclasses=5,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        # stat = Stat()
        replaced = Counter()
        ntok_counter = Counter()
        label_set = Counter()

        def replaced_consumer(word, idx):
            # global stat
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')

            sample = {k: v[0] for k, v in example.items()}
            labels = sample['labels']
            # print(f'{labels.size()}')
            labels = labels.view(-1).numpy().tolist()
            for l in labels:
                label_set.update([l])

            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        print(f'Label set: {label_set}')
        return end_stat


class DPTreeSeparateSST5PlbBinarizerDataset(DPTreeSeparateBinarizerDataset):

    def __init__(self, filename, label_filename, vocab, replaced_consumer, tokenize=tokenize_line, append_eos=False,
                 reverse_order=False, add_if_not_exist=False, noat=False) -> None:
        super().__init__(filename, vocab, replaced_consumer, tokenize, append_eos, reverse_order, add_if_not_exist)
        self.noat = noat
        self.label_filename = label_filename
        self.label_lines = self.retrieve_label_lines()
        assert len(self.data_lines) == len(self.label_lines), f'{len(self.data_lines)} != {len(self.label_lines)}'

    def retrieve_label_lines(self):
        return self._retrieve_lines_from_file(self.label_filename)

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        label_line = self.label_lines[index]
        try:
            example, ntok_ = DPTreeTokenizer.line2multi_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1
            )
        except ValueError as ve:
            print(f'Error Getting Line happened at index {index}')
            raise ve

        try:
            label_example, label_ntok_ = DPTreeTokenizer.line2multi_example(
                s=label_line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                line2example=DPTreeTokenizer.line2example_sst5
            )
        except ValueError as ve:
            print(f'Error Getting Label happened at index {index}')
            raise ve
        assert ntok_ == label_ntok_, f'{ntok_} != {label_ntok_},=== {line} |||| {label_line}'
        example['labels'] = label_example['labels']
        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls,
            filename,
            label_filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            nclasses=5,
            noat=False,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        replaced = Counter()
        ntok_counter = Counter()
        label_set = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, label_filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist, noat=noat,
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')
                print(example)

            sample = {k: v[0] for k, v in example.items()}
            labels = sample['labels']
            labels = labels.view(-1).numpy().tolist()
            for l in labels:
                label_set.update([l])

            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        print(f'Label set: {label_set}')
        return end_stat


class ClassBinarizerDataset(BinarizerDataset):

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            ids = [int(line)]
            assert len(ids) == 1
            ids = torch.tensor(ids).int()
        except Exception as e:
            print(f'cannot int({line})')
            raise e

        return ids

    @classmethod
    def export_binarized_dataset(
            cls, filename,
            vocab,
            consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False, num_workers=0
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, batch in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}]')

            if isinstance(batch, dict):
                sample = {k: v[0] for k, v in batch.items()}
            else:
                sample = batch[0]
            nseq += 1
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': 1, 'replaced': replaced}
        return end_stat


class FloatBinarizerDataset(ClassBinarizerDataset):

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            ids = [float(line)]
            assert len(ids) == 1
            ids = torch.tensor(ids).float()
        except Exception as e:
            print(f'cannot float({line})')
            raise e
        return ids


class NstackTreeSeparateBinarizerDataset(BinarizerDataset):

    def __init__(
            self, filename, vocab, replaced_consumer, tokenize=tokenize_line, append_eos=False,
            reverse_order=False, add_if_not_exist=False,
            remove_root=True, take_pos_tag=True, take_nodes=True
    ) -> None:
        super().__init__(filename, vocab, replaced_consumer, tokenize, append_eos, reverse_order, add_if_not_exist)
        self.remove_root = remove_root
        self.take_pos_tag = take_pos_tag
        self.take_nodes = take_nodes
        print(f'{self.__class__.__name__}::{self.remove_root} - {self.take_pos_tag} -- {self.take_nodes}')

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            example, ntok_ = NstackTreeTokenizer.line2multi_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                remove_root=self.remove_root,
                take_pos_tag=self.take_pos_tag,
                take_nodes=self.take_nodes,
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            raise er

        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls, filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            remove_root=True, take_pos_tag=True, take_nodes=True,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        replaced = Counter()
        ntok_counter = Counter()

        def replaced_consumer(word, idx):
            # global stat
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')
            sample = {k: v[0] for k, v in example.items()}
            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        # ntok = sum(ntok_counter.values())

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        return end_stat


class NstackTreeMergeBinarizerDataset(NstackTreeSeparateBinarizerDataset):

    def __init__(self, filename, vocab, replaced_consumer, tokenize=tokenize_line, append_eos=False,
                 reverse_order=False, add_if_not_exist=False, remove_root=True, take_pos_tag=True,
                 take_nodes=True, reverse_node=True, no_collapse=False, tolower=False) -> None:
        super().__init__(filename, vocab, replaced_consumer, tokenize, append_eos, reverse_order, add_if_not_exist,
                         remove_root, take_pos_tag, take_nodes)
        self.reverse_node = reverse_node
        self.tolower = tolower
        self.no_collapse = no_collapse
        print(f'{self.__class__.__name__}::{self.remove_root} - {self.take_pos_tag} -- {self.take_nodes} -- rvnode{self.reverse_node}')
        print(f'{self.__class__.__name__}::nocolap={self.no_collapse}, lower={self.tolower}')
        self.print_check = False

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        try:
            example, ntok_, nsent = NstackTreeTokenizer.line2multimerge_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                remove_root=self.remove_root,
                take_pos_tag=self.take_pos_tag,
                take_nodes=self.take_nodes,
                reverse_node=self.reverse_node,
                no_collapse=self.no_collapse,
                tolower=self.tolower
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            raise er

        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls, filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            remove_root=True, take_pos_tag=True, take_nodes=True, reverse_node=True,
            no_collapse=False,
            tolower=False,
            **kwargs,
    ):

        nseq, ntok = 0, 0
        replaced = Counter()
        ntok_counter = Counter()

        def replaced_consumer(word, idx):
            # global stat
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node,
            no_collapse=no_collapse,
            tolower=tolower,
        )

        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        cur_time = time.time()
        # load_time = comsune_time = cur_time
        load_time = consume_time = 0
        prev_time = cur_time
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                cur_time = time.time()
                load_time = cur_time - prev_time
                prev_time = cur_time
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}], load_time={load_time}')
                # print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}], load_time={load_time}, consune_time={consume_time}')
                load_time = consume_time = 0
            sample = {k: v[0] for k, v in example.items()}
            nseq += 1
            ntok += ntok_[0].item()

            # ctime = time.time()
            consumer(sample)
            # cur_time = time.time()
            # consume_time += cur_time - ctime
            # load_time += cur_time - prev_time
            # prev_time = cur_time

        # ntok = sum(ntok_counter.values())

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        # print(end_stat)
        return end_stat

    @classmethod
    def export_binarized_separate_dataset_multi_builder(
            cls, filename,
            vocab, consumer,
            tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            num_workers=0,
            remove_root=True, take_pos_tag=True, take_nodes=True, reverse_node=True,
            no_collapse=False,
            tolower=False,
            **kwargs,
    ):
        pass


# NSTACK_KEYS = ['leaves', 'nodes', 'pos_tags', 'spans']
class NstackTreeMergeSST5BinarizerDataset(NstackTreeMergeBinarizerDataset):
    SST5_NSTACK_KEYS = ['leaves', 'nodes', 'pos_tags', 'spans', 'label_leaves', 'label_nodes']

    def __init__(self, filename, label_filename, vocab, replaced_consumer, tokenize=tokenize_line, append_eos=False,
                 reverse_order=False, add_if_not_exist=False, remove_root=True, take_pos_tag=True, take_nodes=True,
                 reverse_node=True) -> None:
        super().__init__(filename, vocab, replaced_consumer, tokenize, append_eos, reverse_order, add_if_not_exist,
                         remove_root, take_pos_tag, take_nodes, reverse_node)
        self.label_filename = label_filename
        self.label_lines = self._retrieve_lines_from_file(self.label_filename)
        assert len(self.data_lines) == len(self.label_lines), f'{len(self.data_lines)} != {len(self.label_lines)}'

    def __getitem__(self, index: int):
        line = self.data_lines[index]
        label_line = self.label_lines[index]
        try:
            example, ntok_, nsent = NstackTreeTokenizer.line2multimerge_example(
                s=line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                remove_root=self.remove_root,
                take_pos_tag=self.take_pos_tag,
                take_nodes=self.take_nodes,
                reverse_node=self.reverse_node,
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            raise er

        try:
            label_example, label_ntok_, nsent_ = NstackTreeTokenizer.line2multimerge_example(
                s=label_line,
                vocab=self.vocab,
                consumer=self.replaced_consumer,
                tokenize=self.tokenize_line,
                append_eos=self.append_eos,
                reverse_order=self.reverse_order,
                add_if_not_exist=self.add_if_not_exist,
                offset=0,
                end=1,
                remove_root=self.remove_root,
                take_pos_tag=self.take_pos_tag,
                take_nodes=self.take_nodes,
                reverse_node=self.reverse_node,
                label_only=True
            )
        except ValueError as ve:
            print(f'Error happened at index {index}')
            raise ve
        except RecursionError as er:
            print('Error retrieving tree recusion')
            raise er
        """
        "leaves": leaves,
        "nodes": nodes,
        """
        snodes_ = example['nodes']
        lnodes_ = label_example['nodes']
        assert snodes_.size() == lnodes_.size(), f'{line} /// {label_line} /// {snodes_} /// {lnodes_}'

        # assert ntok_ == label_ntok_, f'{ntok_} != {label_ntok_}'
        assert ntok_ == label_ntok_, f'{ntok_} != {label_ntok_},=== {line} |||| {label_line}'
        example['label_leaves'] = label_example['pos_tags']
        example['label_nodes'] = label_example['nodes']
        return example, ntok_

    @classmethod
    def export_binarized_separate_dataset(
            cls, filename, label_filename, vocab, consumer, tokenize=tokenize_line, append_eos=False,
            reverse_order=False, add_if_not_exist=False, num_workers=0, remove_root=True,
            take_pos_tag=True, take_nodes=True, reverse_node=True, **kwargs):
        nseq, ntok = 0, 0
        replaced = Counter()
        ntok_counter = Counter()
        label_set = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, label_filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node
        )
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')
                # print(example)

            sample = {k: v[0] for k, v in example.items()}
            label_leaves = sample['label_leaves']
            label_nodes = sample['label_nodes']
            label_leaves = label_leaves.view(-1).numpy().tolist()
            label_nodes = label_nodes.view(-1).numpy().tolist()
            for l in label_leaves:
                label_set.update([l])
            for l in label_nodes:
                label_set.update([l])

            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        print(f'Label set: {label_set}')
        return end_stat


class NstackTreeMergeSST5NodeLMBinarizerDataset(NstackTreeMergeSST5BinarizerDataset):
    @classmethod
    def export_binarized_separate_dataset(
            cls, filename, label_filename, vocab, consumer, tokenize=tokenize_line, append_eos=False,
            reverse_order=False, add_if_not_exist=False, num_workers=0, remove_root=True,
            take_pos_tag=True, take_nodes=True, reverse_node=True, **kwargs):
        nseq, ntok = 0, 0
        replaced = Counter()
        ntok_counter = Counter()
        label_set = Counter()
        node_tgt_set = Counter()

        def replaced_consumer(word, idx):
            if idx == vocab.unk_index and word != vocab.unk_word:
                replaced.update([word])
            ntok_counter.update([word])

        dataset = cls(
            filename, label_filename, vocab, replaced_consumer,
            tokenize, append_eos, reverse_order, add_if_not_exist,
            remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, reverse_node=reverse_node
        )
        dataloader = data.DataLoader(dataset, shuffle=False, batch_size=1, num_workers=num_workers)
        for i, (example, ntok_) in enumerate(dataloader):
            if i == 1 or i % PRINT_INTERVAL == 0:
                now = str(datetime.datetime.now().time())
                print(f'{cls.__name__}:export_separate_dataset:[{now}]: nseq[{nseq}], ntok[{ntok_}]')
                # print(example)

            sample = {k: v[0] for k, v in example.items()}
            label_leaves = sample['label_leaves']
            label_nodes = sample['label_nodes']
            label_leaves = label_leaves.view(-1).numpy().tolist()
            label_nodes = label_nodes.view(-1).numpy().tolist()
            for l in label_leaves:
                label_set.update([l])
            for l in label_nodes:
                label_set.update([l])

            nseq += 1
            ntok += ntok_[0].item()
            consumer(sample)

        end_stat = {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
        print(end_stat)
        print(f'Label set: {label_set}')
        return end_stat




