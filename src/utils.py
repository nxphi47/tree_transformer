from collections import defaultdict, OrderedDict
import importlib.util
import logging
import os
import re
import sys
import traceback

import torch
import torch.nn.functional as F
from torch.serialization import default_restore_location

from fairseq import utils

from torch import nn


# def parse_embedding(embed_path):
#     """Parse embedding text file into a dictionary of word and embedding tensors.
#
#     The first line can have vocabulary size and dimension. The following lines
#     should contain word and embedding separated by spaces.
#
#     Example:
#         2 5
#         the -0.0230 -0.0264  0.0287  0.0171  0.1403
#         at -0.0395 -0.1286  0.0275  0.0254 -0.0932
#     """
#     embed_dict = {}
#     with open(embed_path) as f_embed:
#         next(f_embed)  # skip header
#         for line in f_embed:
#             pieces = line.rstrip().split(" ")
#             embed_dict[pieces[0]] = torch.Tensor([float(weight) for weight in pieces[1:]])
#     return embed_dict


def load_embedding_wmode(embed_dict, vocab, embedding, mode):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
            # nn.init.constant_(embedding.weight[idx], )
    return embedding




