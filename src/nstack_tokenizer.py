from collections import Counter
import os, re

import datetime
import torch
from multiprocessing import Pool
from nltk import Tree
from torch.utils import data
from fairseq.data import Dictionary

from .dptree import tree_process, tree_builder

from .dptree import nstack_process

SPACE_NORMALIZER = re.compile(r"\s+")

PRINT_INTERVAL = int(os.environ.get('PRINT_INTERVAL', 10000))

PLACE_HOLDER = '<placeholder>'

NSTACK_KEYS = ['leaves', 'nodes', 'pos_tags', 'spans']


# "leaves": leave_indices,
#             "nodes": node_indices,
#             "pos_tags": pos_tag_indices,
#             "spans": span_indices

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


class NstackTreeTokenizer(object):

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

    @staticmethod
    def line2example(
            s, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1,
            remove_root=True, take_pos_tag=True, take_nodes=True,
            no_collapse=False,
            label_only=False, tolower=False):
        leaves, pos_tags, nodes, spans = nstack_process.tree_string_to_leave_pos_node_span(
            s, remove_root=remove_root, no_collapse=no_collapse)

        if tolower:
            leaves = ' '.join(leaves).lower().split()
            pos_tags = ' '.join(pos_tags).lower().split()
            nodes = ' '.join(nodes).lower().split()

        leave_indices = NstackTreeTokenizer.tokenize(
            words=leaves,
            vocab=vocab,
            tokenize=tokenize,
            add_if_not_exist=add_if_not_exist,
            consumer=consumer,
            append_eos=append_eos,
            reverse_order=reverse_order,
        )
        if label_only:
            pos_tag_indices = torch.tensor([int(x) for x in pos_tags]).int()
            node_indices = torch.tensor([int(x) for x in nodes]).int()
        else:
            pos_tag_indices = NstackTreeTokenizer.tokenize(
                words=pos_tags,
                vocab=vocab,
                tokenize=tokenize,
                add_if_not_exist=add_if_not_exist,
                consumer=consumer,
                append_eos=append_eos,
                reverse_order=reverse_order,
            )

            node_indices = NstackTreeTokenizer.tokenize(
                words=nodes,
                vocab=vocab,
                tokenize=tokenize,
                add_if_not_exist=add_if_not_exist,
                consumer=consumer,
                append_eos=append_eos,
                reverse_order=reverse_order,
            )

        span_indices = torch.tensor(spans).int()
        assert span_indices.dim() == 2, f'{s}: {leaves}, {pos_tags}, {nodes}, {spans}'
        assert span_indices.size(0) == node_indices.size(0)
        example = {
            "leaves": leave_indices,
            "nodes": node_indices,
            "pos_tags": pos_tag_indices,
            "spans": span_indices
            # "length": line_len
        }
        return example

    @staticmethod
    def line2multi_example(
            s, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1,
            line2example=None,
            # noat=False,
            # cnf=True,
            remove_root=True, take_pos_tag=True, take_nodes=True
    ):
        tree_strings = s.split(nstack_process.NstackTreeBuilder.SENT_SPLITTER)
        try:
            if line2example is None:
                line2example = NstackTreeTokenizer.line2example
            examples = [line2example(
                x, vocab, consumer, tokenize, append_eos, reverse_order, add_if_not_exist,
                offset, end, remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes
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

        ntok = sum([len(x['leaves']) + len(x['nodes']) for x in examples])

        examples_d = {
            "leaves": merge_trees([x['leaves'] for x in examples], 1),
            "nodes": merge_trees([x['nodes'] for x in examples], 1),
            "pos_tags": merge_trees([x['pos_tags'] for x in examples], 1),
            "spans": merge_trees([x['spans'] for x in examples], 0),
            # "length": torch.cat([x['length'] for x in examples], 0),
        }
        return examples_d, ntok

    @staticmethod
    def line2multimerge_example(
            s, vocab, consumer, tokenize=tokenize_line,
            append_eos=False, reverse_order=False,
            add_if_not_exist=False,
            offset=0, end=-1,
            line2example=None,
            remove_root=True, take_pos_tag=True, take_nodes=True,
            reverse_node=True,
            no_collapse=False,
            label_only=False,
            tolower=False
    ):
        # different line2multi_example is that it merge multiple sentences together-> single leave, nodes
        tree_strings = s.split(nstack_process.NstackTreeBuilder.SENT_SPLITTER)

        try:
            if line2example is None:
                line2example = NstackTreeTokenizer.line2example
            examples = [line2example(
                x, vocab, consumer, tokenize, append_eos, reverse_order, add_if_not_exist,
                offset, end, remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
                no_collapse=no_collapse,
                label_only=label_only, tolower=tolower,
            ) for x in tree_strings]
        except ValueError as ve:
            print(f'Error in this example')
            print(s)
            raise ve

        keys = list(examples[0].keys())

        def merge_seq(tensors, pad_idx=1, reverse=False):
            """
                [(n1, d...), (n2, d...), (nm, d...)]
            :param tensors:
            :param pad_idx:
            :return: [sum(n1...nm), d...]
            """
            # total_len = sum(v.size(0) for v in tensors)
            out = torch.cat(tensors, 0)
            if reverse:
                out = torch.flip(out, [0])
            return out

        def merge_indices(tensors, leave_lengths, pad_idx=0, reverse=False):
            """
                [(n1, d...), (n2, d...), (nm, d...)]
            :param tensors:
            :param leave_lengths
            :param pad_idx:
            :return: [sum(n1...nm), d...]
            """
            assert len(tensors) == len(leave_lengths), f'{len(tensors)} != {len(leave_lengths)}'
            agg_tensors = []
            cur_len = 0
            for i, (t, l) in enumerate(zip(tensors, leave_lengths)):
                agg_tensors.append(cur_len + t)
                cur_len += l
            out = torch.cat(agg_tensors, 0)
            if reverse:
                out = torch.flip(out, [0])
            return out

        ntok = sum([len(x['leaves']) + len(x['nodes']) for x in examples])

        leaves_list = [x['leaves'] for x in examples]
        nodes_list = [x['nodes'] for x in examples]
        pos_tags_list = [x['pos_tags'] for x in examples]
        spans_list = [x['spans'] for x in examples]

        leave_lens = [x.size(0) for x in leaves_list]

        leaves = merge_seq(leaves_list, pad_idx=1, reverse=False)
        pos_tags = merge_seq(pos_tags_list, pad_idx=1, reverse=False)
        nodes = merge_seq(nodes_list, pad_idx=1, reverse=reverse_node)
        spans = merge_indices(spans_list, leave_lens, pad_idx=0, reverse=reverse_node)

        nsent = len(leave_lens)

        examples_d = {
            "leaves": leaves,
            "nodes": nodes,
            "pos_tags": pos_tags,
            "spans": spans,
        }
        return examples_d, ntok, nsent

    @staticmethod
    def line2_all_tokens(s, remove_root=True, take_pos_tag=True, take_nodes=True, no_collapse=False, tolower=False):
        token_list = []
        strings = s.split(tree_builder.TreeBuilder.SENT_SPLITTER)
        # print(f'line2_all_tokens: {remove_root}, {take_pos_tag}, {take_nodes}')
        for s in strings:
            try:
                # leaves, pos_tags, nodes, spans = nstack_process.tree_string_to_leave_pos_node_span(
                #     s, remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
                #     no_collapse=no_collapse)
                tokens = nstack_process.tree_string_to_symbols(
                    s,
                    remove_root=remove_root, take_pos_tag=take_pos_tag,
                    take_nodes=take_nodes,
                    no_collapse=no_collapse
                )
            except IndexError as e:
                print(s)
                raise e
            except RecursionError as er:
                print("Recursion error due to too long tree -> omit the tree")
                continue
            if tolower:
                # leaves = ' '.join(leaves).lower().split()
                # pos_tags = ' '.join(pos_tags).lower().split()
                # nodes = ' '.join(nodes).lower().split()
                tokens= ' '.join(tokens).lower().split()

            # token_list += leaves
            # if take_pos_tag:
            #     token_list += pos_tags
            # if take_nodes:
            #     token_list += nodes
            token_list += tokens
        return token_list

    @staticmethod
    def acquire_vocab_multithread(
            filename, vocab, tokenize=tokenize_line, num_workers=1,
            add_single_thread=None, remove_root=True, take_pos_tag=True, take_nodes=True,
            no_collapse=False,
            tolower=False
    ):
        def merge_result(counter):
            for w, c in counter.items():
                vocab.add_symbol(w, c)

        if add_single_thread is None:
            add_single_thread = NstackTreeTokenizer.add_to_vocab_single_thread

        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    add_single_thread,
                    (filename, tokenize, vocab.eos_word, worker_id, num_workers,
                     remove_root, take_pos_tag, take_nodes, no_collapse, tolower)
                ))
            pool.close()
            pool.join()
            assert len(results) == num_workers, f'{len(results)} processes died!'
            for r in results:
                merge_result(r.get())
        else:
            merge_result(
                NstackTreeTokenizer.add_to_vocab_single_thread(
                    filename, tokenize, vocab.eos_word,
                    remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes, no_collapse=no_collapse,
                    tolower=tolower))

    @staticmethod
    def add_to_vocab_single_thread(
            filename, tokenize, eos_word, worker_id=0, num_workers=1,
            remove_root=True, take_pos_tag=True, take_nodes=True, no_collapse=False, tolower=False):
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
                token_list = NstackTreeTokenizer.line2_all_tokens(
                    line, remove_root=remove_root, take_pos_tag=take_pos_tag, take_nodes=take_nodes,
                    no_collapse=no_collapse, tolower=tolower)
                for word in token_list:
                    counter.update([word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    # @staticmethod
    # def binarize_separate(
    #         filename, vocab, consumer, tokenize=tokenize_line,
    #         append_eos=False, reverse_order=False,
    #         add_if_not_exist=False,
    #         offset=0, end=-1,
    #         worker_id=None
    # ):
    #     nseq, ntok = 0, 0
    #     replaced = Counter()
    #
    #     def replaced_consumer(word, idx):
    #         if idx == vocab.unk_index and word != vocab.unk_word:
    #             replaced.update([word])
    #
    #     with open(filename, 'r') as f:
    #         f.seek(offset)
    #         # next(f) breaks f.tell(), hence readline() must be used
    #         line = safe_readline(f)
    #         while line:
    #             if 0 < end < f.tell():
    #                 break
    #
    #             example, ntok_ = NstackTreeTokenizer.line2multi_example(
    #                 s=line,
    #                 vocab=vocab,
    #                 consumer=replaced_consumer,
    #                 tokenize=tokenize,
    #                 append_eos=append_eos,
    #                 reverse_order=reverse_order,
    #                 add_if_not_exist=add_if_not_exist,
    #                 offset=offset,
    #                 end=end
    #             )
    #
    #             nseq += 1
    #             # ntok += sum([len(x) for x in example['nodes']])
    #             ntok += ntok_
    #             consumer(example)
    #             line = f.readline()
    #
    #             if nseq == 1 or nseq % PRINT_INTERVAL == 0:
    #                 now = str(datetime.datetime.now().time())
    #                 print(f'Dptree:binarize:[{now}]-[worker_id={worker_id}] offset={offset}: '
    #                       f'nseq={nseq}, dict={len(vocab)}')
    #
    #     return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}
