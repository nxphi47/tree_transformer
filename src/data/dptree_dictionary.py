
from collections import Counter
from multiprocessing import Pool
import os

import torch

from fairseq.tokenizer import tokenize_line
from fairseq.binarizer import safe_readline
from fairseq.data import data_utils, Dictionary


class DPTreeWrapperDictionary(Dictionary):

    def __init__(self, pad='<pad>', eos='</s>', unk='<unk>', no_strip_node_label=False):
        super().__init__(pad, eos, unk)
        self.no_strip_node_label = no_strip_node_label

    @classmethod
    def load(cls, f, ignore_utf_errors=False, no_strip_node_label=False):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        if isinstance(f, str):
            try:
                if not ignore_utf_errors:
                    with open(f, 'r', encoding='utf-8') as fd:
                        return cls.load(fd)
                else:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fd:
                        return cls.load(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception("Incorrect encoding detected in {}, please "
                                "rebuild the dataset".format(f))

        d = cls(no_strip_node_label=no_strip_node_label)
        lines = f.readlines()
        indices_start_line = d._load_meta(lines)
        for line in lines[indices_start_line:]:
            idx = line.rfind(' ')
            if idx == -1:
                raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
            word = line[:idx]
            count = int(line[idx + 1:])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]
        if self.no_strip_node_label:
            sent = ' '.join(token_string(i) for i in tensor if i != self.eos())
        else:
            sent = ' '.join(token_string(i) for i in tensor if i != self.eos() and "_node_label" not in token_string(i))
        return data_utils.process_bpe_symbol(sent, bpe_symbol)




