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


class CusCoreNLPParser(CoreNLPParser):
    def api_call(self, data, properties=None, timeout=18000000):
        if properties is None:
            properties = {'parse.binaryTrees': "true"}
        return super().api_call(data, properties, timeout)
    @classmethod
    def build_parser(cls, port=9001):
        port = str(port)
        return cls(url=f'http://localhost:{port}')


PARSER_TIMEOUT = int(os.environ.get('PARSER_TIMEOUT', 60000000))
RETRIEVE_BATCH = int(os.environ.get('RETRIEVE_BATCH', 1000))
PARSER_PORT = str(os.environ.get('PARSER_PORT', 9001))
# PARSER_PORT = int(os.environ.get('PARSER_URL', "http://localhost:9001"))

CORENLP_PARSER = None


def get_corenlp_parser():
    global CORENLP_PARSER
    if CORENLP_PARSER is None:
        print(f'!!!! Retrieving CoreNLPParser, please make sure the server is running')
        print(
            f'Server command: java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port {PARSER_PORT} -port {PARSER_PORT} -timeout 15000 & ')
        # CORENLP_PARSER = CusCoreNLPParser(url=f'http://localhost:9001')
        CORENLP_PARSER = CusCoreNLPParser(url=f'http://localhost:{PARSER_PORT}')
    return CORENLP_PARSER


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
                # print(f'Leave reached: {sole_child}')
                pass
            else:
                assert isinstance(sole_child, Tree)
                if sole_child.label() == '@NodeSet':
                    # print(f'Replace @NodeSet, len ={len(sole_child)}')
                    # grand_children = list(sole_child)
                    parent.clear()
                    for i in range(len(sole_child)):
                        parent.append(sole_child[i])
                    # parent.pretty_print()
                    queue_tree.put(parent)
                else:
                    queue_tree.put(sole_child)
        # print('-' * 20  + f' Step {step} ' + '-' * 20)
        step += 1
        # ntree.pretty_print()
    return ntree


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
            if len(child) == 1 and isinstance(child[0], Tree):
                parent.append(child[0])
            else:
                parent.append(child)
        for child in list(parent):
            if isinstance(child, Tree):
                queue_tree.put(child)
        step += 1
    if remove_root and ntree.label() == 'ROOT':
        ntree = ntree[0]
    return ntree


def remove_atnodeset_single_nodeset(tree):
    # todo: remove consecutive single-child nodes, take the last child and remove the parents
    # fixme: somehow still a lot of case it does not work
    ntree = remove_nodeset(tree)
    ntree = remove_single_nodeset(ntree)
    return ntree


def raw_phrase_2_flat(inf, outf):
    with open(inf, 'r') as f:
        tlines = f.read().strip().split('\n\n')
        lines = [' '.join(x.replace('----- getBestPCFGParsePhi(true) -----\n', '').split()) for x in tlines]
        print(f'lines {len(lines)}')
        lines = [' '.join(str(remove_atnodeset_single_nodeset(Tree.fromstring(x))).split()) for x in lines]
    with open(outf, 'w') as f:
        for i, l in enumerate(lines):
            suffix = '' if i == len(lines) - 1 else '\n'
            f.write(f'{l}{suffix}')
    return lines


def raw_phrase_2_flat_v2(inf, outf):
    with open(inf, 'r') as f:
        string = f.read().strip()
        assert "op.nodePrune" not in string
        assert "stripSubcat" not in string
        tlines = string.split('\n\n')
        lines = [' '.join(x.split()) for x in tlines]
        print(f'number of lines {len(lines)}')
        lines = [' '.join(str(remove_atnodeset_single_nodeset(Tree.fromstring(x))).split()) for x in lines]
    with open(outf, 'w') as f:
        for i, l in enumerate(lines):
            # suffix = '' if i == len(lines) - 1 else '\n'
            f.write(f'{l}\n')
    return lines


def padding_leaves(tree):
    leaves_location = [tree.leaf_treeposition(i) for i in range(len(tree.leaves()))]
    for i in range(len(leaves_location)):
        tree[leaves_location[i]] = "{0:03}".format(i) + "||||" + tree[leaves_location[i]]
    for i in range(len(tree.leaves())):
        if len(tree[tree.leaf_treeposition(i)[:-1]]) > 1:
            tree[tree.leaf_treeposition(i)] = Tree(tree[tree.leaf_treeposition(i)[:-1]].label(), [tree.leaves()[i]])


def bft(tree):
    meta = dict()
    list_subtree = list(tree.subtrees())
    lst_tree = []
    lst = []
    queue_tree = queue.Queue()
    queue_tree.put(tree)
    meta[list_subtree.index(tree)] = []
    found_prob = False
    while not queue_tree.empty():
        node = queue_tree.get()
        if len(node) <= 0:
            warnings.warn("[bft]: len(node) <= 0!! will cause error later")
            found_prob = True
            # node =
        lst.append(node)
        lst_tree.append(meta[list_subtree.index(node)])
        for i in range(len(node)):
            child = node[i]
            if isinstance(child, nltk.Tree):
                meta[list_subtree.index(child)] = deepcopy(meta[list_subtree.index(node)])
                meta[list_subtree.index(child)].append(i)
                queue_tree.put(child)
    return lst, lst_tree, meta



def clean_node(tree):
    t3 = deepcopy(tree)
    t3_lst, t3_lst_tree, t3_meta = bft(t3)
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


def generate_data(tree, cnf=True):
    if cnf:
        cnfTree = deepcopy(tree)
        treetransforms.chomsky_normal_form(cnfTree)
        try:
            pad_cnf_tree = deepcopy(cnfTree)
        except RecursionError as e:
            print(f'Error copy tree')
            raise e
        padding_leaves(pad_cnf_tree)
    else:
        pad_cnf_tree = deepcopy(tree)

    bf_tree, bf_lst_tree, bf_meta = bft(pad_cnf_tree)
    input_node = []
    input_label = []
    input_index = []
    leaves_location = [pad_cnf_tree.leaf_treeposition(i) for i in range(len(pad_cnf_tree.leaves()))]
    for i in range(len(bf_lst_tree)):
        if len(bf_tree[i].leaves()) > 1:
            if '|' in bf_tree[i].label():
                input_node.append("SPLIT_NODE_node_label")
                input_label.append("<pad>")
            else:
                input_node.append(bf_tree[i].label() + "_node_label")
                input_label.append('<pad>')
        else:
            input_label.append(bf_tree[i].label() + "_leaf_label")
            try:
                input_node.append(bf_tree[i].leaves()[0][7:])
            except IndexError as e:
                print(f'index {i}, {len(bf_tree)}, {len(bf_lst_tree)}')
                tree.pretty_print()
                # cnfTree.pretty_print()
                print('pad_cnf_tree......')
                pad_cnf_tree.pretty_print()
                print('pad_cnf_tree --- separate.....')
                print(input_node)
                print(f'bf_tree...')
                for x in bf_tree:
                    print(x)
                print(f'bf_lst_tree...')
                for x in bf_lst_tree:
                    print(x)
                print('Searching...')
                print(bf_tree[i - 1])
                print(bf_tree[i])
                print(bf_tree[i + 1])
                print(bf_tree[i].leaves())
                raise e
        first_leaf = deepcopy(bf_lst_tree[i])
        first_leaf.extend(bf_tree[i].leaf_treeposition(0))
        first_leaf = leaves_location.index(tuple(first_leaf))
        last_leaf = first_leaf + len(bf_tree[i].leaves()) - 1
        input_index.append([first_leaf, last_leaf])
    return input_node, input_label, input_index


def generate_data_v2(tree, cnf=True):
    if cnf:
        cnfTree = deepcopy(tree)
        treetransforms.chomsky_normal_form(cnfTree)
        try:
            pad_cnf_tree = deepcopy(cnfTree)
        except RecursionError as e:
            print(f'Error copy tree')
            raise e
        padding_leaves(pad_cnf_tree)
    else:
        pad_cnf_tree = deepcopy(tree)

    bf_tree, bf_lst_tree, bf_meta = bft(pad_cnf_tree)
    input_node = []
    input_label = []
    input_index = []
    leaves_location = [pad_cnf_tree.leaf_treeposition(i) for i in range(len(pad_cnf_tree.leaves()))]
    for i in range(len(bf_lst_tree)):
        if len(bf_tree[i].leaves()) > 1:
            if '|' in bf_tree[i].label():
                input_node.append("SPLIT_NODE_node_label")
                input_label.append("<pad>")
            else:
                input_node.append(bf_tree[i].label() + "_node_label")
                input_label.append('<pad>')
        else:
            input_label.append(bf_tree[i].label() + "_leaf_label")
            try:
                input_node.append(bf_tree[i].leaves()[0][7:])
            except IndexError as e:
                # print(f'index {i}, {len(bf_tree)}, {len(bf_lst_tree)}')
                # tree.pretty_print()
                # # cnfTree.pretty_print()
                # print('pad_cnf_tree......')
                # pad_cnf_tree.pretty_print()
                # print('pad_cnf_tree --- separate.....')
                # print(input_node)
                # print(f'bf_tree...')
                # for x in bf_tree:
                #     print(x)
                # print(f'bf_lst_tree...')
                # for x in bf_lst_tree:
                #     print(x)
                # print('Searching...')
                # print(bf_tree[i - 1])
                # print(bf_tree[i])
                # print(bf_tree[i + 1])
                # print(bf_tree[i].leaves())
                raise e
        first_leaf = deepcopy(bf_lst_tree[i])
        first_leaf.extend(bf_tree[i].leaf_treeposition(0))
        first_leaf = leaves_location.index(tuple(first_leaf))
        last_leaf = first_leaf + len(bf_tree[i].leaves()) - 1
        input_index.append([first_leaf, last_leaf])
    return input_node, input_label, input_index, bf_tree, bf_lst_tree




def check_binary(tree_string):
    tree = Tree.fromstring(tree_string)
    cnfTree = deepcopy(tree)
    treetransforms.chomsky_normal_form(cnfTree)
    original = ' '.join(str(tree).split())
    binary = ' '.join(str(cnfTree).split())
    assert original == binary, f'Different: {original} ####### {binary}'


def tree2matrix(tree, cnf=True):
    if cnf:
        cnf_tree = deepcopy(tree)
        treetransforms.chomsky_normal_form(cnf_tree)
    else:
        cnf_tree = deepcopy(tree)

    node_label = set([])
    leaf_label = set([])
    leaves = cnf_tree.leaves()
    leaves_position = []
    for i in range(len(leaves)):
        leaves_position.append(cnf_tree.leaf_treeposition(i))
    matrix = []
    for i in range(len(leaves)):
        list_i = ['<pad>'] * len(leaves)
        leaf_i = leaves_position[i]
        for k in range(len(leaf_i) - 1, -1, -1):
            if set(leaf_i[k:]) == set([0]):
                tree_at_k = cnf_tree[leaf_i[:k]]
                label_k = tree_at_k.label()
                if k == len(leaf_i) - 1:
                    leaf_label.add(label_k + "_leaf_label")
                else:
                    node_label.add(label_k + "_node_label")
            list_i[i + len(tree_at_k.leaves()) - 1] = label_k
        matrix.append(list_i)
    node_label.add('<pad>')
    leaf_label.add('<pad>')
    node_label.remove('<pad>')
    leaf_label.remove('<pad>')
    return leaves, matrix, node_label, leaf_label


def text2code(vocabulary, node_data, label_data):
    node_ = []
    label_ = []
    for i in range(len(node_data)):
        code_node = [vocabulary[x] for x in node_data[i]]
        node_.append(code_node)
        code_label = [vocabulary[x] for x in label_data[i]]
        label_.append(code_label)
    return node_, label_


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


def test_phrase_sentiment_match(phrase_file, sentiment_file, ):
    with open(phrase_file, 'r') as f:
        phrase_strings = f.read().strip().split('\n')
    with open(sentiment_file, 'r') as f:
        sentiment_strings = f.read().strip().split('\n')

    assert len(phrase_strings) == len(sentiment_strings), f'{len(phrase_strings)} != {len(sentiment_strings)}'

    length_problems = []
    mismatches = []

    def acquire_info(s):
        tree_og = tree_from_string(s)
        pad_leaves_tree = deepcopy(tree_og)
        padding_leaves(pad_leaves_tree)
        cleaned_tree = clean_node(pad_leaves_tree)
        nodes, labels, indices = generate_data(cleaned_tree)

        return tree_og, pad_leaves_tree, cleaned_tree, nodes, labels, indices

    for i, (ps, ss) in enumerate(zip(phrase_strings, sentiment_strings)):
        p_tree_og, p_pad_leaves_tree, p_cleaned_tree, p_nodes, p_labels, p_indices = acquire_info(
            ps)
        s_tree_og, s_pad_leaves_tree, s_cleaned_tree, s_nodes, s_labels, s_indices = acquire_info(
            ss)

        error_mess = None
        try:
            if len(p_nodes) != len(s_nodes):
                error_mess = f'Length problem: [{len(p_nodes)}][{len(s_nodes)}]\n{p_nodes}\n{s_nodes}'
                raise ValueError

            for j, (pw, sw) in enumerate(zip(p_nodes, s_nodes)):
                if pw != sw:
                    try:
                        p_label = int(sw.split('_')[0])
                    except Exception as e:
                        error_mess = f'Match problem: [{len(p_nodes)}][{len(s_nodes)}](id{j}:: {pw} vs {sw})\n{p_nodes}\n{s_nodes}\n'
                        raise e
        except Exception as e:
            print(f'Problem Error at index {i}')
            print(error_mess)
            print(ps)
            print('---' * 10)
            print(ss)
            print('---' * 10)
            p_tree_og.pretty_print()
            print('-' * 10)
            s_tree_og.pretty_print()
            print('=' * 40)
            mismatches.append((i, ps, ss))

    print(f'mismatch: {len(mismatches)}')
    with open(f'{phrase_file}.mismatched.tsv', 'w') as f:
        f.write(f'index\tphrase_tree\tsentiment_tree\n')
        for (i, ps, ss) in mismatches:
            f.write(f'{i}\t{ps}\t{ss}\n')
    with open(f'{phrase_file}.mismatched.phrase', 'w') as f:
        phrases = [str(Tree.fromstring(x[1])) for x in mismatches]
        f.write('\n\n'.join(phrases))
    with open(f'{phrase_file}.mismatched.sentiment', 'w') as f:
        sentiments = [str(Tree.fromstring(x[2])) for x in mismatches]
        f.write('\n\n'.join(sentiments))


def replace_match_phrase_sentiment(phrase_file, sentiment_file, replace_file, output_prefix):
    with open(phrase_file, 'r') as f:
        phrase_strings = f.read().strip().split('\n')
    with open(sentiment_file, 'r') as f:
        sentiment_strings = f.read().strip().split('\n')
    with open(replace_file, 'r') as f:
        replace_strings = f.read().strip().split('\n')

    assert len(phrase_strings) == len(sentiment_strings), f'{len(phrase_strings)} != {len(sentiment_strings)}'

    length_problems = []
    mismatches = []

    final_phrases = []
    final_sentiments = []

    def acquire_info(s):
        tree_og = tree_from_string(s)
        pad_leaves_tree = deepcopy(tree_og)
        padding_leaves(pad_leaves_tree)
        cleaned_tree = clean_node(pad_leaves_tree)
        nodes, labels, indices = generate_data(cleaned_tree)
        return tree_og, pad_leaves_tree, cleaned_tree, nodes, labels, indices

    for i, (ps, ss) in enumerate(zip(phrase_strings, sentiment_strings)):
        p_tree_og, p_pad_leaves_tree, p_cleaned_tree, p_nodes, p_labels, p_indices = acquire_info(
            ps)
        s_tree_og, s_pad_leaves_tree, s_cleaned_tree, s_nodes, s_labels, s_indices = acquire_info(
            ss)
        try:
            if len(p_nodes) != len(s_nodes):
                error_mess = f'Length problem: [{len(p_nodes)}][{len(s_nodes)}]\n{p_nodes}\n{s_nodes}'
                raise ValueError

            for j, (pw, sw) in enumerate(zip(p_nodes, s_nodes)):
                if pw != sw:
                    p_label = int(sw.split('_')[0])

            final_phrases.append(ps)
            final_sentiments.append(ss)
        except Exception as e:
            mismatches.append((i, ps, ss))
            print(f'Replace at index {i}')
            rs = replace_strings[0]
            replace_strings = replace_strings[1:]
            r_tree_og, r_pad_leaves_tree, r_cleaned_tree, r_nodes, r_labels, r_indices = acquire_info(
                rs)
            try:
                if len(r_nodes) != len(s_nodes):
                    error_mess = f'Length problem: [{len(r_nodes)}][{len(s_nodes)}]\n{r_nodes}\n{s_nodes}'
                    raise ValueError
                for j, (rw, sw) in enumerate(zip(r_nodes, s_nodes)):
                    if rw != sw:
                        p_label = int(sw.split('_')[0])

                final_phrases.append(rs)
                final_sentiments.append(ss)
            except Exception as e:
                print(f'Replacement fail')
                print('-----')
                print(ps)
                print('-----')
                print(ss)
                print('-----')
                print(rs)
                print('-----')
                raise e

    print(f'phrase: {len(final_phrases)}')
    print(f'sentiment: {len(final_sentiments)}')
    with open(f'{output_prefix}.phrase', 'w') as f:
        f.write('\n'.join(final_phrases))
    with open(f'{output_prefix}.sentiment', 'w') as f:
        f.write('\n'.join(final_sentiments))





def remove_leaf_label(tree):
    ntree = deepcopy(tree)
    queue_tree = queue.Queue()
    queue_tree.put(ntree)
    step = 0
    while not queue_tree.empty():
        parent = queue_tree.get()
        children = list(parent)
        parent.clear()
        for child in children:
            if isinstance(child, Tree) and len(child) == 1 and isinstance(child[0], str):
                parent.append(child[0])
            else:
                parent.append(child)
        for child in children:
            if isinstance(child, Tree):
                queue_tree.put(child)
    return ntree


def get_tree_encodings(binary_parse):

    binary_parse = binary_parse.replace('(', ' ( ').replace(')', ' ) ')
    sentence = binary_parse.replace('(', ' ').replace(')', ' ')
    # binary_parse = re.sub(r'\(\d', ' ( ', binary_parse).replace(')', ' ) ')
    # sentence = re.sub(r'\(\d', ' ', binary_parse).replace(')', ' ')
    words = sentence.split()
    components = binary_parse.split()
    # print(components)
    final_answers = list()
    stack = list()
    curr_index = 0
    non_leaf_index = len(words)
    for w in components:
        if w == '(':  # guard
            stack.append(w)
        elif w != ')':  # shift
            stack.append(curr_index)
            curr_index += 1
        else:  # reduce
            index_left = stack[-2]
            index_right = stack[-1]
            final_answers.append(index_left)
            final_answers.append(index_right)
            final_answers.append(non_leaf_index)
            stack = stack[:len(stack)-3]
            stack.append(non_leaf_index)
            non_leaf_index += 1
    # print(stack)
    # print(final_answers)
    # assert len(stack) == 1, f'{components} ===== {stack}'
    assert len(stack) == 1
    assert stack[0] == 2 * curr_index - 2
    assert curr_index == len(words)
    final_answers = [str(x) for x in final_answers]
    return ','.join(final_answers)


def build_tree_enconding_dataset(binary_tree_file, out_file=None, out_binary_class=False):
    if out_file is None:
        nclasses = 2 if out_binary_class else 5
        out_file = f'{binary_tree_file}.tree_enc.c{nclasses}.json'

    with open(binary_tree_file, 'r') as f:
        lines = f.read().strip().split('\n')

    def get_label(line):
        tree = Tree.fromstring(line)
        label = tree.label()
        int_label = int(label) + 1

        if out_binary_class:
            if int_label > 3:
                return 2
            elif int_label < 3:
                return 1
            else:
                raise ValueError
        else:
            return int_label

    def extract_information(ori_bin_line):
        # try:
        label = get_label(line)
        sentence = ' '.join(Tree.fromstring(line).flatten())

        parsed = list(get_corenlp_parser().parse_text(sentence))[0]
        cnfTree = deepcopy(parsed)
        treetransforms.chomsky_normal_form(cnfTree)
        pad_cnf_tree = deepcopy(cnfTree)
        padding_leaves(pad_cnf_tree)

        binary_tree = clean_node(pad_cnf_tree)

        # pad_cnf_tree
        tree = remove_atnodeset_single_nodeset(binary_tree)
        tree = remove_leaf_label(tree)
        tree_string = ' '.join(str(tree).split())

        # tree_string_replace = re.sub(r'\(\w+\s', ' ( ', tree_string)
        # NP|<JJ-NN>
        tree_string_replace = re.sub(r'\([\w<>\|\.:\'`$,-]+\s', ' ( ', tree_string)
        # json.loads()
        try:
            tree_enc = get_tree_encodings(tree_string_replace)
        except Exception as e:
            # print(tree.)
            tree.pretty_print()
            # clean_node(tree).pretty_print()
            print(f'get_tree_encodings for line')
            print(line)
            print(tree_string)
            print(tree_string_replace)
            print()
            tree_from_string(line).pretty_print()
            sentiment_tree_enc = get_tree_encodings(line)
            print(sentiment_tree_enc)
            raise e

        obj = {
            "label": label,
            "sentence": sentence,
            "constituency_tree_encoding": tree_enc
        }
        return obj

    # tree_enc_liens = [get_tree_encodings()]
    tree_enc_lines = []
    tree_label = []
    sentences = []
    data = []
    skipped = []
    for i, line in enumerate(lines):
        try:
            obj = extract_information(line)
        except ValueError as e:
            skipped.append(line)
            continue

        data.append(json.dumps(obj))

    print(f'Finished: processed: {len(data)}, skipped: {len(skipped)}')
    # json.dump(data, open(out_file, 'w'))
    with open(out_file, 'w') as f:
        f.write('\n'.join(data))
    print(f'Write to file {out_file}')


def test_no_cnf(file):
    with open(file, 'r') as f:
        lines = f.read().strip().split('\n')

    for i, line in enumerate(lines):
        tree_string = line
        tree_line = tree_from_string(tree_string)

        padding_leaves(tree_line)
        tree_line = clean_node(tree_line)
        # safe_tree = tree_from_string(tree_line)

        line_leaves, line_matrix, line_node_label, line_leaf_label = tree2matrix(tree_line, False)

        # try:
        line_node, line_label, line_index, bf_tree, bf_lst_tree = generate_data_v2(tree_line, False)

        print(f'--- Index {i} ----')
        print(f'nodes[{len(line_node)}]: {line_node}')
        print(f'labels[{len(line_label)}]: {line_label}')
        tree_line.pretty_print()
        print(bf_tree)
        print(bf_lst_tree)

        print('=' * 50)

        input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', default='match_phrase_sentiment')
    parser.add_argument('--phrase_file')
    parser.add_argument('--binary_tree_file')
    parser.add_argument('--replace_phrase_file')
    parser.add_argument('--replace_output_prefix')

    parser.add_argument('--out_binary_class', action='store_true',)
    parser.add_argument('--sentiment_file')

    args = parser.parse_args()

    if args.func == 'match_phrase_sentiment':
        test_phrase_sentiment_match(args.phrase_file, args.sentiment_file)
    elif args.func == 'replace_match_phrase_sentiment':
        replace_match_phrase_sentiment(args.phrase_file, args.sentiment_file, args.replace_phrase_file, args.replace_output_prefix)
    elif args.func == 'get_tree_encodings':
        build_tree_enconding_dataset(args.binary_tree_file, out_binary_class=args.out_binary_class)
    elif args.func == "raw_2_flat":
        raw_phrase_2_flat_v2(args.phrase_file, f'{args.phrase_file}.bin_flattened')
    elif args.func == 'test_no_cnf':
        test_no_cnf(args.phrase_file)
    else:
        raise ValueError(f'invalid func {args.func}')





