from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import nltk
from nltk import treetransforms
from copy import deepcopy
from collections import deque
import queue
from nltk.parse import CoreNLPParser
import argparse
import json
from nltk.tree import Tree
import os
import warnings
import re

import torch
import traceback

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from nltk.tree import Tree
from nltk import tree, treetransforms
import queue

# from bllipparser import RerankingParser
import datetime
# rrp = RerankingParser.from_unified_model_dir("/home/ttnguyen/Downloads/WSJ-PTB3/")
from nltk.parse import CoreNLPParser
from copy import deepcopy
from nltk.parse import CoreNLPDependencyParser
import argparse
from shutil import copyfile
import itertools


RETRIEVE_BATCH = int(os.environ.get('RETRIEVE_BATCH', 1000))
PARSER_TIMEOUT = int(os.environ.get('PARSER_TIMEOUT', 60000000))
PARSER_PORT = str(os.environ.get('PARSER_PORT', 9001))

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


class CusCoreNLPParser(CoreNLPParser):
    def api_call(self, data, properties=None, timeout=18000000, lang=None):
        if properties is None:
            properties = {'parse.binaryTrees': "true"}
        return super().api_call(data, properties, timeout)
    @classmethod
    def build_parser(cls, port=9001):
        port = str(port)
        return cls(url=f'http://localhost:{port}')


def remove_nodeset(tree):
    ntree = deepcopy(tree)
    queue_tree = queue.Queue()
    queue_tree.put(ntree)
    step = 0
    while not queue_tree.empty():
        parent = queue_tree.get()
        if len(parent) > 1:
            for i in range(len(parent)):
                queue_tree.put(parent[i])
        else:
            sole_child = parent[0]
            if isinstance(sole_child, str):
                pass
            else:
                assert isinstance(sole_child, Tree)
                if sole_child.label() == '@NodeSet':
                    parent.clear()
                    for i in range(len(sole_child)):
                        parent.append(sole_child[i])
                    queue_tree.put(parent)
                else:
                    queue_tree.put(sole_child)
        step += 1
    return ntree

parser = CusCoreNLPParser.build_parser(9000)
import time
def f(x, n):
    x = x.lower()
    print(f'[{n}][{len(x.split())}]: {x}')
    all_diff = 0
    for i in range(n):
        before = time.time()
        p = list(parser.parse_text(x))
        after = time.time()
        all_diff += after - before
    # diff = after - before
    print(f'Elapse: {all_diff}')
    return all_diff


def remove_single_nodeset(tree, remove_root=False):
    ntree = deepcopy(tree)
    queue_tree = queue.Queue()
    queue_tree.put(ntree)
    step = 0
    while not queue_tree.empty():
        parent = queue_tree.get()
        children = list(parent)
        parent.clear()
        for child in children:
            # if len(child) == 1 and isinstance(child[0], Tree):
            #     parent.append(child[0])
            # else:
            #     parent.append(child)
            cur_child = child
            while len(cur_child) == 1 and isinstance(cur_child[0], Tree):
                cur_child = cur_child[0]
            parent.append(cur_child)
        for child in list(parent):
            if isinstance(child, Tree):
                queue_tree.put(child)
        step += 1
    if remove_root and ntree.label() == 'ROOT':
        ntree = ntree[0]
    return ntree


def remove_atnodeset_single_nodeset(tree, remove_root=False):
    # todo: remove consecutive single-child nodes, take the last child and remove the parents
    # fixme: somehow still a lot of case it does not work
    ntree = remove_nodeset(tree)
    ntree = remove_single_nodeset(ntree, remove_root)
    return ntree


# from copy import deepcopy
# import queue
# from nltk import Tree
def clean_maybe_rmnode(tree):
    ntree = deepcopy(tree)
    if ntree.label() == "ROOT" and len(ntree) == 1:
        ntree = deepcopy(tree[0])
    while len(ntree) == 1 and isinstance(ntree[0], Tree) and not (len(ntree[0]) == 1 and isinstance(ntree[0][0], str)):
        ntree = ntree[0]
    queue_tree = queue.Queue()
    queue_tree.put(ntree)
    step = 0
    while not queue_tree.empty():
        parent = queue_tree.get()
        children = list(parent)
        parent.clear()
        for child in children:
            cur_child = child
            while len(cur_child) == 1 and isinstance(cur_child[0], Tree):
                cur_child = cur_child[0]
            parent.append(cur_child)
        for child in list(parent):
            if isinstance(child, Tree):
                queue_tree.put(child)
    return ntree


def breadth_first_search(tree):
    """

    :param tree:
    :return:
        tree_node_lst: list of nodes following BFS order
        meta_lst:   not sure
        meta:   not sure
    """
    meta = dict()
    list_subtree = list(tree.subtrees())
    meta_lst = []
    tree_node_lst = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    meta[list_subtree.index(tree)] = []
    found_prob = False
    while not queue_tree.empty():
        node = queue_tree.get()
        if len(node) <= 0:
            warnings.warn("[bft]: len(node) <= 0!! will cause error later")
            found_prob = True
        tree_node_lst.append(node)
        meta_lst.append(meta[list_subtree.index(node)])
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, nltk.Tree):
                meta[list_subtree.index(child)] = deepcopy(meta[list_subtree.index(node)])
                meta[list_subtree.index(child)].append(i)
                queue_tree.put(child)
    return tree_node_lst, meta_lst, meta


def leaves2span(in_leaves, leaves):
    # FIXME: this will cause wrong if the phrase repeat!
    query = in_leaves[0]
    start_idx = [i for (y, i) in zip(leaves, range(len(leaves))) if query == y]
    for idx in start_idx:
        if ' '.join(in_leaves) == ' '.join(leaves[idx:idx + len(in_leaves)]):
            return [idx, idx + len(in_leaves) - 1]
    raise ValueError(f'Not found: {in_leaves}\nIN: {leaves}')


def tree_to_leave_pos_node_span(tree):
    leaves = tree.leaves()
    pos_tags = []
    # meta = dict()
    # list_subtree = list(tree.subtrees())
    # meta_lst = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    # meta[list_subtree.index(tree)] = []
    found_prob = False
    while not queue_tree.empty():
        node = queue_tree.get()
        if len(node) <= 0:
            warnings.warn("[bft]: len(node) <= 0!! will cause error later")
        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue
        tree_node_lst.append(node)
        # meta_lst.append(meta[list_subtree.index(node)])
        # create the spans
        internal_leaves = node.leaves()
        spans.append(leaves2span(internal_leaves, leaves))
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, nltk.Tree):
                # meta[list_subtree.index(child)] = deepcopy(meta[list_subtree.index(node)])
                # meta[list_subtree.index(child)].append(i)
                queue_tree.put(child)
    nodes = [x.label() for x in tree_node_lst]
    return leaves, pos_tags, nodes, spans, tree_node_lst


def tree_to_leave_pos_node_span_collapse(tree):
    # print(f'tree_to_leave_pos_node_span_collapse.....')
    leaves = tree.leaves()
    pos_tags = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue
        while len(node) == 1 and isinstance(node[0], nltk.Tree):
            node.set_label(node[0].label())
            node[0:] = [c for c in node[0]]
        tree_node_lst.append(node)
        internal_leaves = node.leaves()
        spans.append(leaves2span(internal_leaves, leaves))
        for c in node:
            if isinstance(c, nltk.Tree):
                queue_tree.put(c)
    del queue_tree
    nodes = [x.label() for x in tree_node_lst]
    return leaves, pos_tags, nodes, spans, tree_node_lst


def tree_to_leave_pos_node_span_collapse_v2(tree):
    # print(f'tree_to_leave_pos_node_span_collapse.....')
    leaves = tree.leaves()
    len_leave = len(leaves)
    pos_tags = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    level = 0
    start = 0
    end = len_leave - 1
    while not queue_tree.empty():
        node = queue_tree.get()
        while len(node) == 1 and isinstance(node[0], nltk.Tree):
            node.set_label(node[0].label())
            node[0:] = [c for c in node[0]]
        internal_leaves = node.leaves()
        if level == 0:
            _span = [start, len_leave - 1]
            level += 1
        else:
            _span = [start, start + len(internal_leaves) - 1]
            start = start + len(internal_leaves)
            # print(start)
            if start >= len_leave:
                # end
                start = 0
                level += 1
        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue
        tree_node_lst.append(node)
        spans.append(_span)
        # spans.append(leaves2span(internal_leaves, leaves))
        # loc = [t.leaf_treeposition(i) for i in range(3)]
        for c in node:
            if isinstance(c, nltk.Tree):
                queue_tree.put(c)
    del queue_tree
    nodes = [x.label() for x in tree_node_lst]
    print(f'{len(spans)}, {len(nodes)}')
    tree.pretty_print()
    for n, s in zip(nodes, spans):
        print(f'[{n}]: {s}')
    return leaves, pos_tags, nodes, spans, tree_node_lst


def padding_leaves_wnum(leaves, tree):
    # leaves_location = [tree.leaf_treeposition(i) for i in range(len(tree.leaves()))]
    # for i in range(len(leaves_location)):
    #     # tree[leaves_location[i]] = "{0:03}".format(i) + "||||" + tree[leaves_location[i]]
    #     tree[leaves_location[i]] = f'{i}'
    # for i in range(len(tree.leaves())):
    #     if len(tree[tree.leaf_treeposition(i)[:-1]]) > 1:
    #         tree[tree.leaf_treeposition(i)] = Tree(tree[tree.leaf_treeposition(i)[:-1]].label(), [tree.leaves()[i]])
    for i in range(len(leaves)):
        tree[tree.leaf_treeposition(i)] = f'{i}'


def tree_to_leave_pos_node_span_collapse_v3(tree):
    # print(f'tree_to_leave_pos_node_span_collapse.....')
    leaves = tree.leaves()
    # tree.pretty_print()
    # len_leave = len(leaves)
    padding_leaves_wnum(leaves, tree)
    pos_tags = []
    tree_node_lst = []
    spans = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        while len(node) == 1 and isinstance(node[0], nltk.Tree):
            node.set_label(node[0].label())
            node[0:] = [c for c in node[0]]
        if len(node) == 1 and isinstance(node[0], str):
            pos_tags.append(node.label())
            continue
        internal_leaves = node.leaves()
        tree_node_lst.append(node)
        _span = [int(internal_leaves[0]), int(internal_leaves[-1])]
        spans.append(_span)
        # spans.append(leaves2span(internal_leaves, leaves))
        # loc = [t.leaf_treeposition(i) for i in range(3)]
        for c in node:
            if isinstance(c, nltk.Tree):
                queue_tree.put(c)
    del queue_tree
    nodes = [x.label() for x in tree_node_lst]
    if len(nodes) == 0:
        nodes = [tree.label()]
        spans = [[0, len(leaves) - 1]]
    # print(f'{len(spans)}, {len(nodes)}')
    # tree.pretty_print()
    # for n, s in zip(nodes, spans):
    #     print(f'[{n}]: {s}')
    return leaves, pos_tags, nodes, spans, tree_node_lst
# here
# leaves, pos_tags, nodes, spans, tree_node_lst = tree_to_leave_pos_node_span_collapse_v3(t)

def tree_string_to_symbols(tree_string, remove_root=True, no_collapse=False, **kwargs):
    tree = tree_from_string(tree_string)
    # if not no_collapse:
    #     if remove_root:
    #         tree = clean_maybe_rmnode(tree)
    #     else:
    #         # tree = remove_atnodeset_single_nodeset(tree, remove_root=remove_root)
    #         tree = remove_single_nodeset(tree, remove_root)
    leaves = tree.leaves()
    labels = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    while not queue_tree.empty():
        node = queue_tree.get()
        labels.append(node.label())
        if len(node) == 1 and isinstance(node[0], str):
            # node is terminal, its only child is a leaves
            continue
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, nltk.Tree):
                queue_tree.put(child)
    tokens = leaves + labels
    return tokens


def clean_node(tree):
    """
        ### Similar to: remove_atnodeset_single_nodeset
    :param tree:
    :return:
    """
    t3 = deepcopy(tree)
    t3_lst, t3_lst_tree, t3_meta = breadth_first_search(t3)
    for ind, sub in reversed(list(enumerate(t3.subtrees()))):
        if sub.height() >= 2:
            postn = t3_meta[ind]
            parentpos = postn[:-1]
            if parentpos and len(t3[parentpos]) == 1:
                t3[parentpos] = t3[postn]
    leaves_location = [t3.leaf_treeposition(i) for i in range(len(t3.leaves()))]
    for i in range(len(leaves_location)):
        t3[leaves_location[i]] = t3[leaves_location[i]][7:]
    if len(t3) == 1:
        t3 = t3[0]
    return t3


def collapse_unary(tree, collapsePOS=False, collapseRoot=False, joinChar="+"):
    """
    Collapse subtrees with a single child (ie. unary productions)
    into a new non-terminal (Tree node) joined by 'joinChar'.
    This is useful when working with algorithms that do not allow
    unary productions, and completely removing the unary productions
    would require loss of useful information.  The Tree is modified
    directly (since it is passed by reference) and no value is returned.

    :param tree: The Tree to be collapsed
    :type  tree: Tree
    :param collapsePOS: 'False' (default) will not collapse the parent of leaf nodes (ie.
                        Part-of-Speech tags) since they are always unary productions
    :type  collapsePOS: bool
    :param collapseRoot: 'False' (default) will not modify the root production
                         if it is unary.  For the Penn WSJ treebank corpus, this corresponds
                         to the TOP -> productions.
    :type collapseRoot: bool
    :param joinChar: A string used to connect collapsed node values (default = "+")
    :type  joinChar: str
    """

    if collapseRoot == False and isinstance(tree, Tree) and len(tree) == 1:
        nodeList = [tree[0]]
    else:
        nodeList = [tree]

    # depth-first traversal of tree
    while nodeList != []:
        node = nodeList.pop()
        if isinstance(node, Tree):
            if (
                len(node) == 1
                and isinstance(node[0], Tree)
                and (collapsePOS == True or isinstance(node[0, 0], Tree))
            ):
                node.set_label(node.label() + joinChar + node[0].label())
                node[0:] = [child for child in node[0]]
                # since we assigned the child's children to the current node,
                # evaluate the current node again
                nodeList.append(node)
            else:
                for child in node:
                    nodeList.append(child)


def collapse_unary_last(tree, collapsePOS=False, collapseRoot=False, joinChar="+"):
    """
    Collapse subtrees with a single child (ie. unary productions)
    into a new non-terminal (Tree node) joined by 'joinChar'.
    This is useful when working with algorithms that do not allow
    unary productions, and completely removing the unary productions
    would require loss of useful information.  The Tree is modified
    directly (since it is passed by reference) and no value is returned.

    :param tree: The Tree to be collapsed
    :type  tree: Tree
    :param collapsePOS: 'False' (default) will not collapse the parent of leaf nodes (ie.
                        Part-of-Speech tags) since they are always unary productions
    :type  collapsePOS: bool
    :param collapseRoot: 'False' (default) will not modify the root production
                         if it is unary.  For the Penn WSJ treebank corpus, this corresponds
                         to the TOP -> productions.
    :type collapseRoot: bool
    :param joinChar: A string used to connect collapsed node values (default = "+")
    :type  joinChar: str
    """

    if not collapseRoot and isinstance(tree, Tree) and len(tree) == 1:
        nodeList = [tree[0]]
    else:
        nodeList = [tree]

    # depth-first traversal of tree
    while nodeList != []:
        node = nodeList.pop()
        if isinstance(node, Tree):
            if (
                len(node) == 1
                and isinstance(node[0], Tree)
                and (collapsePOS or isinstance(node[0, 0], Tree))
            ):
                # node.set_label(node.label() + joinChar + node[0].label())
                node.set_label(node[0].label())
                node[0:] = [child for child in node[0]]
                # since we assigned the child's children to the current node,
                # evaluate the current node again
                nodeList.append(node)
            else:
                for child in node:
                    nodeList.append(child)


def padding_leaves(tree):
    leaves_location = [tree.leaf_treeposition(i) for i in range(len(tree.leaves()))]
    for i in range(len(leaves_location)):
        tree[leaves_location[i]] = "{0:03}".format(i) + "||||" + tree[leaves_location[i]]
    for i in range(len(tree.leaves())):
        if len(tree[tree.leaf_treeposition(i)[:-1]]) > 1:
            tree[tree.leaf_treeposition(i)] = Tree(tree[tree.leaf_treeposition(i)[:-1]].label(), [tree.leaves()[i]])


def tree_str_post_process(tree_string):
    tree_string = tree_string.replace('-LRB- (', '-LRB- -LRB-').replace('-RRB- )', '-RRB- -RRB-')
    tree_string = tree_string.replace('TRUNC (', 'TRUNC -LRB-').replace('TRUNC )', 'TRUNC -RRB-')
    return tree_string


def tree_from_string(tree_string):
    try:
        s = tree_string
        s = tree_str_post_process(s)
        tree = Tree.fromstring(s)
    except Exception as e:
        # print(f'Tree.fromstring(tree_string) failed, try to omit the post_process')
        try:
            tree = Tree.fromstring(tree_string)
        except Exception as e:
            print(f'ERROR: unable to parse the tree')
            print(tree_string)
            raise e
    return tree


def tree_string_to_leave_pos_node_span(tree_string, remove_root=True, no_collapse=False, **kwargs):
    tree = tree_from_string(tree_string)
    if not no_collapse:
        # if remove_root:
        #     tree = clean_maybe_rmnode(tree)
        # else:
        # backhere
        # tree = remove_atnodeset_single_nodeset(tree, remove_root=remove_root)
        # tree = collapse_unary_last(tree)
        # leaves, pos_tags, nodes, spans, tree_node_lst = tree_to_leave_pos_node_span_collapse(tree)
        leaves, pos_tags, nodes, spans, tree_node_lst = tree_to_leave_pos_node_span_collapse_v3(tree)
        # leaves, pos_tags, nodes, spans, tree_node_lst = tree_to_leave_pos_node_span_collapse_v2(tree)
    else:
        leaves, pos_tags, nodes, spans, tree_node_lst = tree_to_leave_pos_node_span(tree)
    return leaves, pos_tags, nodes, spans




# TODO: Nstack Dataset builder ---------------------
def replace_special_character(string):
    new_string = deepcopy(string)
    # new_string = new_string.replace(u"）", ")").replace(u"（", "(")
    new_string = new_string.replace(")", u"）").replace("(", u"（")

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


def remap_chars(tree):
    for i in range(len(tree.leaves())):
        if tree.leaves()[i] in SPECIAL_CHAR_MBACK:
            tree[tree.leaf_treeposition(i)] = SPECIAL_CHAR_MBACK[tree.leaves()[i]]


def parse_string(parser, bpe_string, unify_tree=True):
    word_string_nobpe = bpe_string.replace("@@ ", "")
    word_string = replace_special_character(word_string_nobpe)
    # tree_strings = list(get_corenlp_parser().parse_text(word_string, timeout=PARSER_TIMEOUT))
    try:
        tree_strings = list(parser.parse_text(word_string))
    except Exception as e:

        try:
            print(f'Try bpe version')
            tree_strings = list(parser.parse_text(bpe_string))
        except Exception as ee:
            print(f'Failed.')
            print(f'[Ori]: {bpe_string}')
            print(f'[Proc]: {word_string}')
            traceback.print_stack()
            raise ee
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


# fixme: temp: imdb preprocessing
def proc(x, y):
    with open(x, 'r') as f:
        s = f.read().replace('< br />< br />', '')
    with open(y, 'w') as f:
        f.write(s)

# proc('dev.input', 'dev.proc.input')

def max_tokens(x):
    with open(x, 'r') as f:
        longest = max(len(x.split(" ")) for x in f.read().strip().split('\n'))
        print(longest)


class NStackDataset(Dataset):
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
        self.parser = CusCoreNLPParser(url=f'http://localhost:{self.port}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].rstrip("\n")
        # if self.transform:
        try:
            parse_strings, _, token_string_set = parse_string(self.parser, sample, unify_tree=self.unify_tree)
        except Exception as e:
            print(f'Error happen at index {idx}, return empty')
            print(f'{sample}')
            # raise e
            return {
                'ori_sample': sample,
                'pstring': [],
                'token_set': []
            }

        sample = {
            'ori_sample': sample,
            'pstring': parse_strings,
            'token_set': list(token_string_set)
        }
        return sample


class Nstack2SeqDataset(Dataset):
    def __init__(self, src_file, tgt_file, transform=None, unify_tree=True, port=PARSER_PORT):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.src_data = open(src_file, "r").readlines()
        self.tgt_data = open(tgt_file, "r").readlines()
        assert len(self.src_data) == len(self.tgt_data), f'{len(self.src_data)} != {len(self.tgt_data)}'

        print(f'Finish reading lines [port={port}]: {len(self.src_data)}')
        print(f'[init]Line 0: {self.src_data[0]} --> {self.tgt_data[0]}')
        self.transform = transform
        self.port = port
        self.unify_tree = unify_tree
        self.parser = CusCoreNLPParser(url=f'http://localhost:{self.port}')

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_sample = self.src_data[idx].rstrip("\n")
        tgt_sample = self.tgt_data[idx].rstrip("\n")
        # if self.transform:
        try:
            parse_strings, _, token_string_set = parse_string(self.parser, src_sample, unify_tree=self.unify_tree)
        except Exception as e:
            print(f'Error happen at index {idx}, return empty')
            print(f'{src_sample}')
            print(f'{tgt_sample}')
            print(traceback.format_exc())
            # raise e
            return {
                'ori_sample': src_sample,
                'pstring': [],
                'token_set': [],
                'tgt': ''
            }

        src_sample = {
            'ori_sample': src_sample,
            'pstring': parse_strings,
            'token_set': list(token_string_set),
            'tgt': tgt_sample,
        }
        return src_sample


class BuildTree2LeavesDataset(Dataset):

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tree = tree_from_string(self.data[index])
        leaves = tree.leaves()
        return " ".join(leaves)


class NstackTreeBuilder(object):
    SENT_SPLITTER = '#####------#####'

    def __init__(self, transform=None, unify_tree=True, bpe_tree=False, bpe_code=None, ignore_error=True) -> None:
        super().__init__()
        self.transform = transform
        self.bpe_tree = bpe_tree
        self.bpe_code = bpe_code
        self.unify_tree = unify_tree
        self.ignore_error = ignore_error

        if self.bpe_tree:
            assert bpe_code is not None and os.path.exists(bpe_code)
        print(f'bpe_tree= {self.bpe_tree}')
        print(f'ignore_error= {self.ignore_error}')
        print(f'bpe_code= {self.bpe_code}')

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
                    mess = 'skip it' if self.ignore_error else 'raise error'
                    print(f'WARNING: Failed step {i_batch} -> {mess}')
                    print(sample_batched)
                    print(f'[Sample]: {sample_batched["ori_sample"]}')
                    if self.ignore_error:
                        continue
                    else:
                        raise e

            v = sample_batched['token_set'][0]

            if i_batch % RETRIEVE_BATCH == 0:
                print(f'Retrieve batch [{i_batch}]: time {datetime.datetime.now()}]')

            data.append(s)
            vocab = vocab.union(set(v))

        vocab = list(vocab)
        return data, vocab

    def build_bpe_tree_vocab(self, raw_vocab_file_or_set, output_file):
        # raw_vocab_file = f'{output_file}.raw.vocab'
        bpe_vocab_file = f'{output_file}.bpe.vocab'

        if isinstance(raw_vocab_file_or_set, str):
            assert os.path.exists(raw_vocab_file_or_set)
            is_file = True
            raw_vocab_file = raw_vocab_file_or_set
        else:
            assert isinstance(raw_vocab_file_or_set, (set, list))
            is_file = False
            raw_vocab = list(raw_vocab_file_or_set)
            raw_vocab_file = f"temp.{len(raw_vocab)}.vocab"
            with open(raw_vocab_file, 'w') as f:
                f.write('\n'.join(raw_vocab))

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

    def export_seq_file(self, data, file, separate=True):
        with open(file, "w") as f:
            for i, w in enumerate(data):
                if separate:
                    assert isinstance(w, (list, tuple))
                    s = self.__class__.SENT_SPLITTER.join(w)
                    f.write(f'{s}\n')
                else:
                    f.write(f'{w}\n')
            f.close()

    def read_separate_file(self, file):
        with open(file, 'r') as f:
            lines = f.read().strip().split('\n')
            data = [x.split(self.__class__.SENT_SPLITTER) for x in lines]

        return data

    def apply_bpe_on_tree_strings(self, data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe, multiple_sents=False, workers=0):
        parse_string_data = []
        splitter = self.__class__.SENT_SPLITTER
        class BpeToTreeStringDataset(Dataset):

            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                data_p = self.data[index]
                data_p = data_p if isinstance(data_p, list) else [data_p]
                bpe_trees = []
                for x in data_p:
                    try:
                        tree_i = tree_from_string(x)
                    except Exception as e:
                        print(f'---- Error when parse tree at index {index}')
                        raise e
                    new_tree_j = deepcopy(tree_i)
                    leaves = tree_i.leaves()
                    for j, word in enumerate(leaves):
                        if word in word2bpe and len(word2bpe[word]) > 1:
                            # reassign
                            loc_leaf_j = tree_i.leaf_treeposition(j)
                            # eg: (0, 1, 0)...
                            pos_tag_j = tree_i[loc_leaf_j[:-1]].label()
                            bpe_tree_j = Tree(
                                pos_tag_j,
                                [Tree(f'{pos_tag_j}_bpe', [x]) for x in word2bpe[word]]
                            )
                            new_tree_j[loc_leaf_j[:-1]] = bpe_tree_j
                    parsing_bpe_tree = ' '.join(str(new_tree_j).split())
                    bpe_trees.append(parsing_bpe_tree)
                merged = splitter.join(bpe_trees)
                return merged

            def __init__(self, data) -> None:
                super().__init__()
                self.data = data

        def parse(x, index=0):
            try:
                tree_i = tree_from_string(x)
            except Exception as e:
                print(f'---- Error when parse tree at index {index}')
                raise e
            new_tree_j = deepcopy(tree_i)
            leaves = tree_i.leaves()
            for j, word in enumerate(leaves):
                if word in word2bpe and len(word2bpe[word]) > 1:
                    # reassign
                    loc_leaf_j = tree_i.leaf_treeposition(j)
                    # eg: (0, 1, 0)...
                    pos_tag_j = tree_i[loc_leaf_j[:-1]].label()
                    bpe_tree_j = Tree(
                        pos_tag_j,
                        [Tree(f'{pos_tag_j}_bpe', [x]) for x in word2bpe[word]]
                    )
                    new_tree_j[loc_leaf_j[:-1]] = bpe_tree_j
                    # try:
                    #     new_tree_j[loc_leaf_j[:-1]] = bpe_tree_j
                    #     new_tree_j[loc_leaf_j[:-2]] = bpe_tree_j
                    # except Exception as e:
                    #     print(f'index={index}, ')

                    # new_tree_j[loc_leaf_j[:-2]] = bpe_tree_j
            parsing_bpe_tree = ' '.join(str(new_tree_j).split())
            return parsing_bpe_tree

        # for i, d in enumerate(data):
        #     if i % 100000 == 0:
        #         print(f'apply_bpe_on_tree_strings:: {i}')
        #     bpes = [parse(x, i) for x in d] if isinstance(d, list) else parse(d, index=i)
        #     parse_string_data.append(bpes)
        dataset = BpeToTreeStringDataset(data)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
        for i, d in enumerate(dataloader):
            if i == 0:
                print(f'First: {d}')
            if i % 200000 == 0:
                print(f'apply_bpe_on_tree_strings:: [{i}]: {d}')

            # bpes = [parse(x, i) for x in d] if isinstance(d, list) else parse(d, index=i)
            parse_string_data.append(d[0])

        return parse_string_data

    def build_bpe_tree(self, data, raw_vocab_file_or_set, before_bpe_output_file, bpe_output_file, separate=True):
        print(f'Proceed to generate bpe tree [separate={separate}]: {bpe_output_file}')
        assert separate
        bpe_vocab_file, word2bpe, list_dict, vocab_bpe = self.build_bpe_tree_vocab(
            raw_vocab_file_or_set, before_bpe_output_file)

        print(f'apply_bpe_on_tree_strings [separate={separate}]: {bpe_output_file}')
        parse_string_data = self.apply_bpe_on_tree_strings(
            data, bpe_vocab_file, word2bpe, list_dict, vocab_bpe
        )
        # self.export_seq_file(parse_string_data, bpe_output_file, separate)
        with open(bpe_output_file, "w") as f:
            # for i, w in enumerate(parse_string_data):
            #     f.write(f'{w}\n')
            f.write('\n'.join(parse_string_data))
            f.close()

    def export_text_to_tree_strings(
            self, input_file, output_file, separate=True, num_workers=0, ignore_if_exist=False,
            port=PARSER_PORT, remove_root=True, tgt_file=None, tgt_out_file=None):

        raw_vocab_file = f'{output_file}.raw.vocab'
        before_bpe_output_file = f'{output_file}.before-bpe'

        if ignore_if_exist:
            if self.bpe_tree and os.path.exists(before_bpe_output_file):
                assert os.path.exists(raw_vocab_file)
                print(f'Ignore tree generation, proceed to tree production')
                bpe_output_file = output_file
                data = self.read_separate_file(before_bpe_output_file)
                self.build_bpe_tree(
                    data, raw_vocab_file, before_bpe_output_file, bpe_output_file, separate=separate)
                return

        nstack_dataset = NStackDataset(input_file, self.transform, not separate, port)
        dataloader = DataLoader(nstack_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        data, vocab = self.retrieve_tree_data(dataloader)
        print(f'Generating raw vocab [separate={separate}]: {raw_vocab_file}')
        self.export_seq_file(vocab, raw_vocab_file, separate=False)
        print(f'Generate tree string data [separate={separate}]: {output_file}')
        self.export_seq_file(data, output_file, separate)

        if self.bpe_tree:
            copyfile(output_file, before_bpe_output_file)
            bpe_output_file = output_file
            self.build_bpe_tree(data, raw_vocab_file, before_bpe_output_file, bpe_output_file, separate=separate)

    def export_bpe_tree_from_nonbpe(
            self, before_bpe_file, after_bpe_file, raw_vocab_file, separate=True,
            tgt_file=None, tgt_out_file=None, workers=0
    ):
        if self.bpe_tree and os.path.exists(before_bpe_file):

            # assert os.path.exists(raw_vocab_file)
            print(f'Ignore tree generation, proceed to tree production')
            print(f'Generating BPE Tree.....')
            bpe_output_file = after_bpe_file
            data = self.read_separate_file(before_bpe_file)
            print(f'Finish reading separate file: {len(data)}')
            # asdasd
            if raw_vocab_file is None or not os.path.exists(raw_vocab_file):
                print(f'raw_vocab not found, rebuilding them!!')
                print(f'get the trees, worker = {workers}')
                # vocab = set()
                # for i, d in enumerate(data):
                #     if i % 20000 == 0:
                #         print(f'raw: {i}')
                #     for t in d:
                #         tree = tree_from_string(t)
                #         leaves = tree.leaves()
                #         vocab = vocab.union(set(leaves))
                # raw_vocab_file = vocab
                # print(f'vocab: {len(raw_vocab_file)}')

                # vocab = set()
                all_leaves = []
                flat_data = list(itertools.chain(*data))
                dataset = BuildTree2LeavesDataset(flat_data)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers)
                for i, d in enumerate(dataloader):
                    if i % 200000 == 0:
                        print(f'raw: {i}, leaves={len(all_leaves)}')
                    if i == 0:
                        print(f'First from raw:{d}')
                    leaves = d[0].split()
                    all_leaves.extend(leaves)
                vocab = set(all_leaves)
                raw_vocab_file = vocab
                print(f'vocab: {len(raw_vocab_file)}')

            self.build_bpe_tree(
                data, raw_vocab_file, before_bpe_file, bpe_output_file, separate=separate)

            print(f'Finish, copying target file: {tgt_file} -> {tgt_out_file}')
            if os.path.exists(tgt_file) and tgt_out_file is not None:
                copyfile(tgt_file, tgt_out_file)


class Nstack2SeqTreeBuilder(NstackTreeBuilder):
    def retrieve_tree_data(self, dataloader):
        src_data = []
        tgt_data = []
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
                    mess = 'skip it' if self.ignore_error else 'raise error'
                    print(f'WARNING: Failed step {i_batch} -> {mess}')
                    print(sample_batched)
                    print(f'[Sample]: {sample_batched["ori_sample"]}')
                    if self.ignore_error:
                        continue
                    else:
                        raise e

            v = sample_batched['token_set'][0]
            tgt = sample_batched['tgt'][0]

            if i_batch % RETRIEVE_BATCH == 0:
                print(f'Retrieve batch [{i_batch}]: time {datetime.datetime.now()}]')

            src_data.append(s)
            tgt_data.append(tgt)
            vocab = vocab.union(set(v))

        vocab = list(vocab)
        return src_data, tgt_data, vocab

    def export_text_to_tree_strings(self, input_file, output_file, separate=True, num_workers=0, ignore_if_exist=False,
                                    port=PARSER_PORT, remove_root=True, tgt_file=None, tgt_out_file=None):
        assert tgt_file is not None and os.path.exists(tgt_file)
        assert tgt_out_file is not None and tgt_out_file != ''

        raw_vocab_file = f'{output_file}.raw.vocab'
        before_bpe_output_file = f'{output_file}.before-bpe'
        print(f'Processing file ${input_file}')
        print(f'Output to {output_file}')

        if ignore_if_exist:
            raise NotImplementedError

        nstack2seq_dataset = Nstack2SeqDataset(input_file, tgt_file, self.transform, not separate, port)
        dataloader = DataLoader(nstack2seq_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
        src_data, tgt_data, vocab = self.retrieve_tree_data(dataloader)

        print(f'Generating raw vocab [separate={separate}]: {raw_vocab_file}')
        self.export_seq_file(vocab, raw_vocab_file, separate=False)
        print(f'Generate tree string data [separate={separate}]: {output_file}')
        self.export_seq_file(src_data, before_bpe_output_file if self.bpe_tree else output_file, separate)
        print(f'Generate tgt data [separate={separate}]: {tgt_out_file}')
        self.export_seq_file(tgt_data, tgt_out_file, False)

        if self.bpe_tree:
            print(f'Start building BPE tree')
            self.build_bpe_tree(src_data, vocab, before_bpe_output_file, output_file, separate=separate)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before')
    parser.add_argument('--after')
    parser.add_argument('--before_tgt')
    parser.add_argument('--after_tgt')

    parser.add_argument('--bpe_code')
    parser.add_argument('--raw_vocab', default=None)
    parser.add_argument('--bpe_tree', action='store_true')
    parser.add_argument('--ignore_error', action='store_true')

    parser.add_argument('--convert_bpe', action='store_true')

    parser.add_argument('--parse_src_only', action='store_true')


    parser.add_argument('--workers', default=0, type=int)
    args = parser.parse_args()

    # , default = 'bert-base-uncased')

    if args.convert_bpe:
        print(f'Convert to BPE form')
        builder = Nstack2SeqTreeBuilder(
            transform=True,
            bpe_tree=args.bpe_tree,
            bpe_code=args.bpe_code,
            unify_tree=False,
            ignore_error=args.ignore_error
        )
        builder.export_bpe_tree_from_nonbpe(args.before, args.after, args.raw_vocab, separate=True,
            tgt_file=args.before_tgt, tgt_out_file=args.after_tgt, workers=args.workers)
        exit()

    if args.parse_src_only:
        print(f'Parse source onlyy')
        builder = NstackTreeBuilder(
            transform=True, bpe_tree=args.bpe_tree, bpe_code=args.bpe_code, unify_tree=False
        )
        builder.export_text_to_tree_strings(
            args.before, args.after, separate=True, remove_root=True, num_workers=0, port=PARSER_PORT,
        )
        # builder.export_bpe_tree_from_nonbpe(
        #     args.before, args.after, args.raw_vocab, separate=True,
        #     tgt_file=args.before_tgt, tgt_out_file=args.after_tgt
        # )
        #
        exit()

    print(f'starting translation')

    # todo: convert bpe from non-bpe
    # builder = NstackTreeBuilder(
    #     transform=True, bpe_tree=args.bpe_tree, bpe_code=args.bpe_code, unify_tree=False
    # )
    # builder.export_bpe_tree_from_nonbpe(
    #     args.before, args.after, args.raw_vocab, separate=True,
    #     tgt_file=args.before_tgt, tgt_out_file=args.after_tgt
    # )

    # todo: parse it again
    builder = Nstack2SeqTreeBuilder(
        transform=True,
        bpe_tree=args.bpe_tree,
        bpe_code=args.bpe_code,
        unify_tree=False,
        ignore_error=args.ignore_error
    )
    builder.export_text_to_tree_strings(
        args.before, args.after, separate=True, remove_root=True, num_workers=0, port=PARSER_PORT,
        ignore_if_exist=False, tgt_file=args.before_tgt, tgt_out_file=args.after_tgt
    )







