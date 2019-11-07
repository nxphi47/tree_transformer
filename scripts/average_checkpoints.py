#!/usr/bin/env python3

import argparse
import collections
import torch
import os
import re
from fairseq.utils import import_user_module


def default_avg_params(params_dict):
    averaged_params = collections.OrderedDict()

    # v should be a list of torch Tensor.
    for k, v in params_dict.items():
        summed_v = None
        for x in v:
            summed_v = summed_v + x if summed_v is not None else x
        averaged_params[k] = summed_v / len(v)

    return averaged_params


def ema_avg_params(params_dict, ema_decay):
    averaged_params = collections.OrderedDict()
    lens = [len(v) for k, v in params_dict.items()]
    assert all(x == lens[0] for x in lens), f'lens params: {lens}'
    num_checkpoints = lens[0]
    # y = x

    for k, v in params_dict.items():
        # order: newest to oldest
        # reverse the order
        # y_t = x_t * decay + y_{t-1} * (1 - decay)
        total_v = None
        for x in reversed(v):
            if total_v is None:
                total_v = x
            else:
                total_v = x * ema_decay + total_v * (1.0 - ema_decay)

        averaged_params[k] = total_v
    return averaged_params


def average_checkpoints(inputs, ema_decay=1.0):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    for i, f in enumerate(inputs):
        state = torch.load(
            f,
            map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            if k not in params_dict:
                params_dict[k] = []
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            params_dict[k].append(p)

    if ema_decay < 1.0:
        print(f'Exponential moving averaging, decay={ema_decay}')
        averaged_params = ema_avg_params(params_dict, ema_decay)
    else:
        print(f'Default averaging')
        averaged_params = default_avg_params(params_dict)
    new_state['model'] = averaged_params
    return new_state


def last_n_checkpoints(paths, n, update_based, upper_bound=None):
    assert len(paths) == 1
    path = paths[0]
    if update_based:
        pt_regexp = re.compile(r'checkpoint_\d+_(\d+)\.pt')
    else:
        pt_regexp = re.compile(r'checkpoint(\d+)\.pt')
    files = os.listdir(path)

    entries = []
    for f in files:
        m = pt_regexp.fullmatch(f)
        if m is not None:
            sort_key = int(m.group(1))
            if upper_bound is None or sort_key <= upper_bound:
                entries.append((sort_key, m.group(0)))
    if len(entries) < n:
        raise Exception('Found {} checkpoint files but need at least {}', len(entries), n)
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)[:n]]


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')
    num_group = parser.add_mutually_exclusive_group()
    num_group.add_argument('--num-epoch-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    num_group.add_argument('--num-update-checkpoints', type=int,
                           help='if set, will try to find checkpoints with names checkpoint_ee_xx.pt in the path specified by input, '
                           'and average last this many of them.')
    parser.add_argument('--checkpoint-upper-bound', type=int,
                        help='when using --num-epoch-checkpoints, this will set an upper bound on which checkpoint to use, '
                        'e.g., with --num-epoch-checkpoints=10 --checkpoint-upper-bound=50, checkpoints 41-50 would be averaged.')

    # parser.add_argument('--ema', type=float, default=1.0, help='exponential moving average decay')
    # parser.add_argument('--no-progress-bar', action='store_true', help='disable progress bar')
    parser.add_argument('--ema', default='False', type=str, metavar='BOOL', help='ema')
    parser.add_argument('--ema_decay', type=float, default=1.0, help='exponential moving average decay')
    parser.add_argument('--user-dir', default=None)

    # fmt: on
    args = parser.parse_args()

    import_user_module(args)
    print(args)

    num = None
    is_update_based = False
    if args.num_update_checkpoints is not None:
        num = args.num_update_checkpoints
        is_update_based = True
    elif args.num_epoch_checkpoints is not None:
        num = args.num_epoch_checkpoints

    assert args.checkpoint_upper_bound is None or args.num_epoch_checkpoints is not None, \
            '--checkpoint-upper-bound requires --num-epoch-checkpoints'
    assert args.num_epoch_checkpoints is None or args.num_update_checkpoints is None, \
            'Cannot combine --num-epoch-checkpoints and --num-update-checkpoints'

    if num is not None:
        args.inputs = last_n_checkpoints(
            args.inputs, num, is_update_based, upper_bound=args.checkpoint_upper_bound,
        )
        # print('averaging checkpoints: ', args.inputs)
        print('averaging checkpoints: ')
        for checkpoint in args.inputs:
            print(checkpoint)
        print('-' * 40)

    # ema = args.ema
    # assert isinstance(args.ema, bool)
    print(f'Start averaing with ema={args.ema}, ema_decay={args.ema_decay}')
    new_state = average_checkpoints(args.inputs, args.ema_decay)
    torch.save(new_state, args.output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()
