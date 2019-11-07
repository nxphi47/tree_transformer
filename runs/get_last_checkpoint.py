import argparse
import collections
import torch
import os
import re


def last_n_checkpoint_index(paths, n, update_based, upper_bound=None):
    # assert len(paths) == 1
    # path = paths[0]
    # assert len(paths) == 1
    path = paths

    if update_based:
        pt_regexp = re.compile(r'checkpoint_\d+_(\d+)\.pt')
    else:
        pt_regexp = re.compile(r'checkpoint(\d+)\.pt')
    files = os.listdir(path)
    # print(files)
    # print(pt_regexp)
    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        # print(paths)
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    last_checkpoints = [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]
    last_checkpoint_index = [x[0] for x in sorted(entries, reverse=True)[:n]][0]
    return last_checkpoint_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', required=True, help='Input checkpoint file paths.')
    args = parser.parse_args()

    assert os.path.exists(args.dir)
    last_checkpoint_index = last_n_checkpoint_index(
        args.dir, 1, False, upper_bound=None,
    )
    print(last_checkpoint_index)

