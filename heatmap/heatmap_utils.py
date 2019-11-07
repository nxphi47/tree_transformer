from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from shutil import copyfile
import os
try:
    import matplotlib

    matplotlib.use('Agg')

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.ticker import PercentFormatter

except Exception as e:
    print(f'matplotlib not found!')
    # raise e

import torch
import pprint

IMAGE_DECODE_LENGTH = 100

DPI = 80
FONT_MULTIPLIER = 3
CELL_FONT_MULTIPLIER = 2
FIG_MULTIPLIER = 2.0


def plot_attention_image(
        title,
        image,
        row_names=None,
        col_names=None,
        out_file=None,
        add_values=True,
        font_multiplier=FONT_MULTIPLIER,
        cell_multiplier=CELL_FONT_MULTIPLIER,
        show=False):
    figsize = np.round(np.array(image.T.shape) * FIG_MULTIPLIER).astype(np.int32).tolist()
    fig, ax = plt.subplots(
        dpi=DPI,
        figsize=figsize
        # frameon=False
    )

    fontdict = {
        "size": font_multiplier * image.shape[0]
    }
    cell_fontdict = {
        "size": cell_multiplier * image.shape[0]
    }

    im = ax.imshow(image)
    # im = ax.imshow(image, cmap='YlGn')
    if row_names is not None:
        ax.set_yticks(np.arange(len(row_names)))
        ax.set_yticklabels(row_names, fontdict=fontdict)
    if col_names is not None:
        ax.set_xticks(np.arange(len(col_names)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(col_names, fontdict=fontdict)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    len_row = image.shape[0]
    len_col = image.shape[1]
    if add_values:
        for i in range(len_row):
            for j in range(len_col):
                text = ax.text(
                    j, i, "%.2f " % image[i, j], ha="center", va="center", color="w", fontdict=cell_fontdict)

    if len(title) > 100:
        title = title[:100] + "-\n" + title[100:]
    ax.set_title(title, fontdict=fontdict)
    fig.tight_layout()

    # plt.show()

    if out_file is not None:
        # plt.savefig(out_file, dpi=16 * (image.shape[0]))
        print(f'out: {out_file}')
        plt.savefig(out_file)

        # out_file_npz = "{}.npz".format(out_file)
        # np.savez(out_file_npz, image=image, cols=np.array(col_names), rows=np.array(row_names))


def save_merge_attention(hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    if torch.is_tensor(hypo_tokens):
        hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    if torch.is_tensor(src_tokens):
        src_tokens = src_tokens.cpu().numpy().tolist()

    assert isinstance(hypo_tokens, list)
    assert isinstance(src_tokens, list)

    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'[{index}] Attention shape: {attention.shape}, hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}], src_tokens({len(src_tokens)})=[{src_tokens}]')
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'Image {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_merge_attention_v2(hypo_str, src_str, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    # if torch.is_tensor(hypo_tokens):
    #     hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    # if torch.is_tensor(src_tokens):
    #     src_tokens = src_tokens.cpu().numpy().tolist()

    # assert isinstance(hypo_tokens, list)
    # assert isinstance(src_tokens, list)
    hypo_tokens = hypo_str.split()
    src_tokens = src_str.split()
    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'Image {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def token_string(dictionary, i, escape_unk=False):
    if i == dictionary.unk():
        return dictionary.unk_string(escape_unk)
    else:
        return dictionary[i]


def idx2tokens(tensor, dictionary):
    if torch.is_tensor(tensor) and tensor.dim() == 2:
        raise NotImplementedError(f'{tensor}')
    tokens = [token_string(dictionary, i) for i in tensor]
    return tokens

# def len_list_tensor(tensor):
#     if torch.is_tensor(tensor):
#         return


def save_merge_attention_v3(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()

    # if torch.is_tensor(hypo_tokens):
    #     hypo_tokens = hypo_tokens.cpu().numpy().tolist()
    # if torch.is_tensor(src_tokens):
    #     src_tokens = src_tokens.cpu().numpy().tolist()

    # assert isinstance(hypo_tokens, list)
    # assert isinstance(src_tokens, list)
    # hypo_tokens = hypo_str.split()
    # src_tokens = src_str.split()
    hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)

    rows = hypo_tokens
    cols = src_tokens + hypo_tokens
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'{model.__class__.__name__} {index}'
    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_default_attention(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    # title = f'{merge_transformer.MergeTransformerModel.__class__.__name__} {index}'
    title = f'{model.__class__.__name__} {index}'

    shape = attention.shape
    rows = hypo_tokens
    cols = src_tokens
    assert shape[0] == len(rows), f'{shape}, {len(rows)}'
    try:
        assert shape[1] == len(cols), f'{shape}, {len(cols)}'
    except AssertionError as e:
        attention = attention[:, :-1]
        shape = attention.shape
        print(e)
        assert shape[1] == len(cols), f'again: {shape}, {len(cols)}'

    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_nstack_attention(model, leaves, nodes, indices, attention, index, save_dir, src_dict=None, tgt_dict=None):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()

    leave_tokens = idx2tokens(leaves, tgt_dict)

    nodes_tokens = idx2tokens(nodes, tgt_dict)
    nodes_tokens = [f'{x}&'.replace('_node_label', f'({leave_tokens[idx[0]]}~{leave_tokens[idx[1]]})') for x, idx in zip(nodes_tokens, indices)]

    concat_tokens = leave_tokens + nodes_tokens
    # if '<pad>&'

    hypo_tokens = src_tokens = concat_tokens

    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'{model.__class__.__name__} {index}'

    shape = attention.shape
    rows = hypo_tokens
    cols = src_tokens
    assert shape[0] == len(rows), f'{shape}, {len(rows)}'
    try:
        assert shape[1] == len(cols), f'{shape}, {len(cols)}'
    except AssertionError as e:
        attention = attention[:, :-1]
        shape = attention.shape
        print(e)
        assert shape[1] == len(cols), f'again: {shape}, {len(cols)}'

    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )


def save_separate_nstack_attention(model, leaves, pos, nodes, indices, attention, index, save_dir, src_dict=None, tgt_dict=None, scale=100.0):
    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()

    leave_len = len(leaves)
    node_len = len(nodes)
    leave_tokens = idx2tokens(leaves, tgt_dict)

    nodes_tokens = idx2tokens(nodes, tgt_dict)

    tree_node_tokens = [x.replace('_node_label', '') for x in nodes_tokens]
    nodes_tokens = [f'{x}&'.replace('_node_label', f'({leave_tokens[idx[0]]}~{leave_tokens[idx[1]]})') for x, idx in zip(nodes_tokens, indices)]

    pos_tokens = idx2tokens(pos, tgt_dict)
    pos_tokens = [x.replace('<pad>', '||') for x in pos_tokens]

    # filter pad in nodes
    if '<pad>' in nodes_tokens[0]:
        node_len = node_len - 1
        nodes_tokens = nodes_tokens[1:]
        tree_node_tokens = tree_node_tokens[1:]
        indices = indices[1:]

    tree_leave_tokens = leave_tokens
    tree_pos_tokens = pos_tokens
    # filter pad in leaves
    if '<pad>' in tree_leave_tokens[-1]:
        tree_leave_tokens = tree_leave_tokens[:-1]
        tree_pos_tokens = tree_pos_tokens[:-1]
    tree_leave_len = len(tree_leave_tokens)

    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'leave_tokens({len(leave_tokens)})=[{leave_tokens}]')
    print(f'nodes_tokens({len(nodes_tokens)})=[{nodes_tokens}]')
    print(f'tree_node_tokens({len(tree_node_tokens)})=[{tree_node_tokens}]')
    print(f'tree_leave_tokens({len(tree_leave_tokens)})=[{tree_leave_tokens}]')
    print(f'indices({len(indices)})=[{indices}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)
    sep_dir = os.path.join(save_dir, 'separate')
    image_dir = os.path.join(sep_dir, f'index_{index:05d}')
    os.makedirs(sep_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    title = f'{model.__class__.__name__} {index}'
    (leave_file, leave_title) = os.path.join(image_dir, f'leave.png'), f'Leave:{model.__class__.__name__} {index}'
    (node_file, node_title) = os.path.join(image_dir, f'node.png'), f'node:{model.__class__.__name__} {index}'

    leave_self_att = attention[:leave_len, :leave_len]

    plot_attention_image(
        leave_title, leave_self_att,
        row_names=leave_tokens,
        col_names=leave_tokens,
        out_file=leave_file
    )

    # todo: for nodes--------------
    # node_on_leaves_nodes_att = attention[-node_len:]

    node_on_nodes_att = attention[-node_len:, -node_len:]
    node_on_leaves_att = attention[-node_len:, :tree_leave_len]

    node_on_nodes_att = node_on_nodes_att * scale
    node_on_leaves_att = node_on_leaves_att * scale

    sep_node_att = []
    for i, (node, indx) in enumerate(zip(tree_node_tokens, indices)):
        non = node_on_nodes_att[i]
        nol = node_on_leaves_att[i]

        node_reprs = []
        for j, _node in enumerate(tree_node_tokens):
            att_value = non[j]

            rep = f'+***{_node}***+_{{{att_value:.2f}}}' if i == j else f'{_node}_{{{att_value:.2f}}}'
            node_reprs.append(rep)

        leave_reprs = []
        pos_reprs = []
        for j, (_leave, _pos) in enumerate(zip(tree_leave_tokens, tree_pos_tokens)):
            att_value = nol[j]
            rep_leave = f'{_leave}_{{{att_value:.2f}}}'
            rep_pos = f'{_pos}_{{{att_value:.2f}}}'
            leave_reprs.append(rep_leave)
            pos_reprs.append(rep_pos)

        leave_brackets = [f'({pos} {leave})' for pos, leave in zip(pos_reprs, leave_reprs)]

        for j, (idx, _node) in enumerate(zip(indices, node_reprs)):
            leave_brackets[idx[0]] = f"({_node} {leave_brackets[idx[0]]}"
            leave_brackets[idx[1]] = f"{leave_brackets[idx[1]]})"

        tree_string = " ".join(leave_brackets)
        rsyntax_string = tree_string.replace('(', "[").replace(")", "]")
        print(f'==>[{node}]: {rsyntax_string}')

        node_folder = os.path.join(image_dir, 'tmp_node_folder')
        os.makedirs(node_folder, exist_ok=True)
        os.system(f'rsyntaxtree -o {node_folder} "{rsyntax_string}"')
        # fixme: https://github.com/yohasebe/rsyntaxtree

        # expect the file png is there
        rsyntax_f_file = os.path.join(node_folder, 'syntree.png')
        node_file = os.path.join(image_dir, f'{i:03d}_node-{node}.png')
        copyfile(rsyntax_f_file, node_file)


def save_attention_by_models(models, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
    assert len(models) == 1, f'models only support 1'
    model = models[0]

    # if isinstance(model, merge_transformer.MergeTransformerModel):
    if model.__class__.__name__ == 'MergeTransformerModel':
        print(f'heat_map_by MergeTransformerModel')
        save_merge_attention_v3(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict, tgt_dict)
    else:
        print(f'heat_map_by Default')
        save_default_attention(model, hypo_tokens, src_tokens, attention, index, save_dir, src_dict, tgt_dict)


# def save_self_attention_nstack(models, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None):
def save_self_attention_nstack(
        models, leaves, pos, nodes, indices, attention, index, save_dir, src_dict=None, tgt_dict=None, img_type='default'):
    assert len(models) == 1, f'models only support 1'
    model = models[0]
    if img_type == 'default':
        save_nstack_attention(model, leaves, nodes, indices, attention, index, save_dir, src_dict, tgt_dict)
    elif img_type == 'separate':
        save_separate_nstack_attention(model, leaves, pos, nodes, indices, attention, index, save_dir, src_dict, tgt_dict)
    else:
        raise NotImplementedError


def save_cross_attention_nstack(
    models, nseq, seqlen, node_idx, hypo_tokens, src_tokens, attention, index, save_dir, src_dict=None, tgt_dict=None, net_input=None):
    assert len(models) == 1, f'models only support 1'
    model = models[0]

    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    if torch.is_tensor(node_idx):
        node_idx = node_idx.cpu().numpy().tolist()

    hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)

    assert len(node_idx) == len(src_tokens)
    # src_tokens = [x if y[0] == y[1] else for x, y in zip(src_tokens, node_idx)]

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    src_tokens_chunks = list(chunks(src_tokens, seqlen))
    node_idx_chunks = list(chunks(node_idx, seqlen))
    o_src_tokens = []
    try:
        for i, (ct, ctn) in enumerate(zip(src_tokens_chunks, node_idx_chunks)):
            oct = [x if y[0] == y[1] else f'{x}<{ct[y[0]]}~{ct[y[1]]}>' for x, y in zip(ct, ctn)]
            o_src_tokens += oct
    except Exception as e:
        print(nseq)
        print(seqlen)
        print(node_idx)
        print(src_tokens)
        print(src_tokens_chunks)
        print(node_idx_chunks)
        raise e

    ori_src_tokens = src_tokens
    src_tokens = o_src_tokens
    print(f'----- index {index}')
    print(f'[{index}] Attention shape: {attention.shape}')
    print(f'hypo_tokens({len(hypo_tokens)})=[{hypo_tokens}]')
    print(f'src_tokens({len(src_tokens)})=[{src_tokens}]')
    print(f'ori_src_tokens({len(ori_src_tokens)})=[{ori_src_tokens}]')
    print('-' * 20)
    os.makedirs(save_dir, exist_ok=True)

    image_file = os.path.join(save_dir, f'{index}.png')
    # title = f'{merge_transformer.MergeTransformerModel.__class__.__name__} {index}'
    title = f'{model.__class__.__name__} {index}'

    shape = attention.shape
    rows = hypo_tokens
    cols = src_tokens
    assert shape[0] == len(rows), f'{shape}, {len(rows)}'
    try:
        assert shape[1] == len(cols), f'{shape}, {len(cols)}'
    except AssertionError as e:
        attention = attention[:, :-1]
        shape = attention.shape
        print(e)
        assert shape[1] == len(cols), f'again: {shape}, {len(cols)}'

    plot_attention_image(
        title, attention,
        row_names=rows,
        col_names=cols,
        out_file=image_file
    )




def save_agg_srcdict_histogram(models, src_agg_att, src_tokens, attention, save_dir, src_dict, tgt_dict):
    assert len(models) == 1, f'models only support 1'
    model = models[0]

    if torch.is_tensor(attention):
        attention = attention.cpu().numpy()
    # hypo_tokens = idx2tokens(hypo_tokens, tgt_dict)
    src_tokens = idx2tokens(src_tokens, src_dict)
    shape = attention.shape
    len_rows = shape[0]

    for i, token in enumerate(src_tokens):
        att = attention[:, i].tolist()
        src_agg_att[token] += att

    title = f'{model.__class__.__name__} Src histogram'

    positive_src_tokens = {k: v for k, v in src_agg_att.items() if len(v) > 0}

    positive_src_tokens = {k: sum(v) / len(v) for k, v in positive_src_tokens.items()}

    hist_tokens = [k for k in positive_src_tokens.keys()]
    hist_values = [v for k, v in positive_src_tokens.items()]
    index = np.arange(len(hist_tokens))
    y = np.array(hist_values)
    labels = hist_tokens

    # N_points = 100000

    # Generate a normal distribution, center at x=0 and y=5
    # x = np.random.randn(N_points)
    # y = .4 * x + np.random.randn(100000) + 5

    fig, ax = plt.subplots(dpi=DPI,
        figsize=(30, 30))

    # index = np.arange(n_groups)
    bar_width = 0.1

    opacity = 0.8
    # error_config = {'ecolor': '0.3'}

    rects1 = ax.bar(index, y, bar_width, alpha=opacity, color='b', label='Hist')

    # rects2 = ax.bar(index + bar_width, means_women, bar_width,
    #                 alpha=opacity, color='r',
    #                 yerr=std_women, error_kw=error_config,
    #                 label='Women')

    ax.set_xlabel('Token')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    # ax.set_xticks(index + bar_width / 2, rotation='vertical')
    # ax.set_xticklabels(labels)

    plt.xticks(index + bar_width / 2, labels, rotation='vertical')
    ax.legend()

    fig.tight_layout()
    # plt.show()
    out_file = os.path.join(save_dir, f'src_hist.png')
    plt.savefig(out_file)

    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # plt.hist()
    # # We can set the number of bins with the `bins` kwarg
    # axs[0].hist(x, bins=n_bins)
    # axs[1].hist(y, bins=n_bins)

