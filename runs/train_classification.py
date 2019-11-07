#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import itertools
import os
import math
import random
import time

import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
# from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from fairseq.utils import import_user_module


from collections import OrderedDict
from itertools import chain

from nltk import Tree as nltkTree

import torch

from fairseq import distributed_utils, models, optim, utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler


class FiTrainer(object):
    """Main class for data parallel training.

    This class supports synchronous distributed data parallel training,
    where multiple workers each have a full model replica and gradients
    are accumulated across workers before each update. We use
    :class:`~torch.nn.parallel.DistributedDataParallel` to handle
    communication of the gradients across workers.
    """

    def __init__(self, args, task, model, criterion, dummy_batch, oom_batch=None):
        self.args = args
        self.task = task

        # copy model and criterion to current device
        self.criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._model = self._model.half()
        if self.cuda:
            self.criterion = self.criterion.cuda()
            self._model = self._model.cuda()

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_model = None

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        return self._wrapped_model

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, params)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            utils.save_state(
                filename, self.args, self.get_model().state_dict(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = utils.load_model_state(
            filename, self.get_model(),
        )
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state

    def train_step(self, samples, dummy_batch=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            try:
                if self.args.distributed_world_size > 1:
                    # Whenever *samples* contains more than one mini-batch, we
                    # want to accumulate gradients locally and only call
                    # all-reduce in the last backwards pass. Currently the
                    # *need_reduction* flag is only supported by
                    # LegacyDistributedDataParallel.
                    if i < len(samples) - 1:
                        self.model.accumulate_grads = True
                    else:
                        self.model.accumulate_grads = False

                # forward and backward
                loss, sample_size, logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer,
                    ignore_grad
                )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(('| WARNING: ran out of memory with exception: {};\n Skipping batch').format(str(e)))
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_sizes, ooms, prev_norms = \
                zip(*distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm],
                ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)
            assert all(norm == prev_norms[0] for norm in prev_norms), \
                'Fatal error: gradients are inconsistent between workers'

        self.meters['oom'].update(ooms, len(samples))
        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.criterion
        )
        sample_size = self.task.grad_denom(sample_sizes, self.criterion)

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))

        try:
            # normalize grads by sample size
            self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            self.optimizer.step()
            self._num_updates += 1

            # update learning rate
            self.lr_scheduler.step_update(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    if self.cuda:
                        torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.criterion
        )
        sample_size = self.task.grad_denom(
            sample_size, self.criterion
        )

        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        if self.cuda:
            sample = self.__class__.move_to_cuda(sample)
        return sample

    @classmethod
    def move_to_cuda(cls, sample):
        if len(sample) == 0:
            return {}

        def _move_to_cuda(maybe_tensor):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.cuda()
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_cuda(value)
                    for key, value in maybe_tensor.items()
                }
            elif isinstance(maybe_tensor, nltkTree):
                return maybe_tensor
            elif isinstance(maybe_tensor, list):
                return [_move_to_cuda(x) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_cuda(sample)

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)


class ModelParallelTrainer(object):
    def __init__(self, args, task, model, criterion, dummy_batch, oom_batch=None, gpu_idx=None):
        self.args = args
        self.task = task
        assert isinstance(gpu_idx, list)
        self.gpu_idx = gpu_idx

        # copy model and criterion to current device
        self.criterion = criterion
        self._model = model
        self.cuda = torch.cuda.is_available() and not args.cpu
        if args.fp16:
            self._model = self._model.half()

        if self.cuda:
            self.criterion = self.criterion.cuda(self.gpu_idx[-1])
            self._model = self._model.setup_cuda(self.gpu_idx)

        self._dummy_batch = dummy_batch
        self._oom_batch = oom_batch

        self._lr_scheduler = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        self._prev_grad_norm = None
        self._wrapped_model = None

        assert self.args.distributed_world_size == 1

        self.init_meters(args)

    def init_meters(self, args):
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        if args.fp16:
            self.meters['loss_scale'] = AverageMeter()  # dynamic loss scale
        self.meters['wall'] = TimeMeter()      # wall time in seconds
        self.meters['train_wall'] = StopwatchMeter()  # train wall time in seconds

    @property
    def model(self):
        if self._wrapped_model is None:
            if self.args.distributed_world_size > 1:
                self._wrapped_model = models.DistributedFairseqModel(
                    self.args, self._model,
                )
            else:
                self._wrapped_model = self._model
        assert self._wrapped_model is not None
        return self._wrapped_model

    @property
    def optimizer(self):
        # print(f'Build optimizer....')
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if self._lr_scheduler is None:
            self._build_optimizer()  # this will initialize self._lr_scheduler
        return self._lr_scheduler

    def _build_optimizer(self):
        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self._optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self._optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self._optimizer = optim.build_optimizer(self.args, params)

        # We should initialize the learning rate scheduler immediately after
        # building the optimizer, so that the initial learning rate is set.
        self._lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        if distributed_utils.is_master(self.args):  # only save one checkpoint
            extra_state['train_meters'] = self.meters
            utils.save_state(
                filename, self.args, self.get_model().state_dict(), self.criterion, self.optimizer,
                self.lr_scheduler, self._num_updates, self._optim_history, extra_state,
            )

    def load_checkpoint(self, filename, reset_optimizer=False, reset_lr_scheduler=False, optimizer_overrides=None):
        """Load all training state from a checkpoint file."""
        extra_state, self._optim_history, last_optim_state = utils.load_model_state(
            filename, self.get_model(),
        )
        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert last_optim['criterion_name'] == self.criterion.__class__.__name__, \
                'criterion does not match; please reset the optimizer (--reset-optimizer)'
            assert last_optim['optimizer_name'] == self.optimizer.__class__.__name__, \
                'optimizer does not match; please reset the optimizer (--reset-optimizer)'

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim['lr_scheduler_state'])
            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self._num_updates = last_optim['num_updates']

        if extra_state is not None and 'train_meters' in extra_state:
            self.meters.update(extra_state['train_meters'])
            del extra_state['train_meters']

            # reset TimeMeters, since their start times don't make sense anymore
            for meter in self.meters.values():
                if isinstance(meter, TimeMeter):
                    meter.reset()

        return extra_state

    def train_step(self, samples, dummy_batch=False):
        """Do forward, backward and parameter update."""
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.zero_grad()

        if not dummy_batch:
            self.meters['train_wall'].start()

        # forward and backward pass
        logging_outputs, sample_sizes, ooms = [], [], 0
        for i, sample in enumerate(samples):
            sample = self._prepare_sample(sample)
            if sample is None:
                # when sample is None, run forward/backward on a dummy batch
                # and ignore the resulting gradients
                sample = self._prepare_sample(self._dummy_batch)
                ignore_grad = True
            else:
                ignore_grad = False

            try:
                assert self.args.distributed_world_size == 1
                if self.args.distributed_world_size > 1:
                    # Whenever *samples* contains more than one mini-batch, we
                    # want to accumulate gradients locally and only call
                    # all-reduce in the last backwards pass. Currently the
                    # *need_reduction* flag is only supported by
                    # LegacyDistributedDataParallel.
                    if i < len(samples) - 1:
                        self.model.accumulate_grads = True
                    else:
                        self.model.accumulate_grads = False

                # forward and backward
                # print(f'Attempt update 1')
                loss, sample_size, logging_output = self.task.train_step(
                    sample, self.model, self.criterion, self.optimizer,
                    ignore_grad
                )

                if not ignore_grad:
                    logging_outputs.append(logging_output)
                    sample_sizes.append(sample_size)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(('| WARNING: ran out of memory with exception: {};\n Skipping batch').format(str(e)))
                    ooms += 1
                    self.zero_grad()
                else:
                    raise e

        if ooms > 0 and self._oom_batch is not None:
            self.handle_ooms(ooms)

        if dummy_batch:
            return None

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_outputs, sample_sizes, ooms, prev_norms = \
                zip(*distributed_utils.all_gather_list(
                    [logging_outputs, sample_sizes, ooms, self._prev_grad_norm],
                ))
            logging_outputs = list(chain.from_iterable(logging_outputs))
            sample_sizes = list(chain.from_iterable(sample_sizes))
            ooms = sum(ooms)
            assert all(norm == prev_norms[0] for norm in prev_norms), \
                'Fatal error: gradients are inconsistent between workers'

        self.meters['oom'].update(ooms, len(samples))
        if ooms == self.args.distributed_world_size * len(samples):
            print('| WARNING: OOM in all workers, skipping update')
            self.zero_grad()
            return None

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_outputs, self.criterion
        )
        sample_size = self.task.grad_denom(sample_sizes, self.criterion)

        if not all(k in logging_output for k in ['ntokens', 'nsentences']):
            raise Exception((
                'Please update the {}.aggregate_logging_outputs() method to '
                'return ntokens and nsentences'
            ).format(self.task.__class__.__name__))
        # print(f'Attempt update 2')
        # backhere
        try:
            # normalize grads by sample size
            self.optimizer.multiply_grads(self.args.distributed_world_size / float(sample_size))

            # clip grads
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)
            self._prev_grad_norm = grad_norm

            # take an optimization step
            self.optimizer.step()
            self._num_updates += 1

            # update learning rate
            self.lr_scheduler.step_update(self._num_updates)

            # update meters
            ntokens = logging_output.get('ntokens', 0)
            nsentences = logging_output.get('nsentences', 0)
            self.meters['wps'].update(ntokens)
            self.meters['ups'].update(1.)
            self.meters['wpb'].update(ntokens)
            self.meters['bsz'].update(nsentences)
            self.meters['gnorm'].update(grad_norm)
            self.meters['clip'].update(
                1. if grad_norm > self.args.clip_norm and self.args.clip_norm > 0 else 0.
            )
            self.meters['train_loss'].update(logging_output.get('loss', 0), sample_size)
            if 'nll_loss' in logging_output:
                self.meters['train_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)
        except OverflowError as e:
            print('| WARNING: overflow detected, ' + str(e))
            self.zero_grad()
            logging_output = None

        if self.args.fp16:
            self.meters['loss_scale'].reset()
            self.meters['loss_scale'].update(self.optimizer.scaler.loss_scale)

        self.meters['train_wall'].stop()

        return logging_output

    def valid_step(self, sample, raise_oom=False):
        """Do forward pass in evaluation mode."""
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample)
            if sample is None:
                sample = self._prepare_sample(self._dummy_batch)
                ignore_results = True
            else:
                ignore_results = False

            try:
                _loss, sample_size, logging_output = self.task.valid_step(
                    sample, self.model, self.criterion
                )
            except RuntimeError as e:
                if 'out of memory' in str(e) and not raise_oom:
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    if self.cuda:
                        torch.cuda.empty_cache()
                    return self.valid_step(sample, raise_oom=True)
                else:
                    raise e

            if ignore_results:
                logging_output, sample_size = {}, 0

        # gather logging outputs from all replicas
        if self.args.distributed_world_size > 1:
            logging_output, sample_size = zip(*distributed_utils.all_gather_list(
                [logging_output, sample_size],
            ))
            logging_output = list(logging_output)
            sample_size = list(sample_size)
        else:
            logging_output = [logging_output]
            sample_size = [sample_size]

        # aggregate logging outputs and sample sizes
        logging_output = self.task.aggregate_logging_outputs(
            logging_output, self.criterion
        )
        sample_size = self.task.grad_denom(
            sample_size, self.criterion
        )

        # update meters for validation
        ntokens = logging_output.get('ntokens', 0)
        self.meters['valid_loss'].update(logging_output.get('loss', 0), sample_size)
        if 'nll_loss' in logging_output:
            self.meters['valid_nll_loss'].update(logging_output.get('nll_loss', 0), ntokens)

        return logging_output

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, dummy_batch=True)
        self.zero_grad()

    def handle_ooms(self, number_of_ooms):
        """
        c10d accumulates/syncs gradients between gpus during backward pass.
        In case of OOMs, gpus may fail to sync, so we manually iterate
        extra to make sure each gpu makes same number of iterations.
        """
        for _ in range(number_of_ooms):
            self.train_step([self._oom_batch], True)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_step(self, epoch, val_loss=None):
        """Adjust the learning rate based on the validation loss."""
        return self.lr_scheduler.step(epoch, val_loss)

    def lr_step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.lr_scheduler.step_update(num_updates)

    def get_lr(self):
        """Get the current learning rate."""
        return self.optimizer.get_lr()

    def get_model(self):
        """Get the (non-wrapped) model instance."""
        return self._model

    def get_meter(self, name):
        """Get a specific meter by name."""
        if name not in self.meters:
            return None
        return self.meters[name]

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        if self.cuda:
            # self.gpu_idx
            sample = self.__class__.move_to_cuda(sample, key_gpu={'target': self.gpu_idx[-1]})
        return sample

    @classmethod
    def move_to_cuda(cls, sample, key_gpu=None):
        if len(sample) == 0:
            return {}

        def _move_to_cuda(maybe_tensor, gpu_id=0):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.cuda(gpu_id)
            elif isinstance(maybe_tensor, dict):
                new_sample = {}
                for key, value in maybe_tensor.items():
                    if key in key_gpu:
                        new_sample[key] = _move_to_cuda(value, key_gpu[key])
                    else:
                        new_sample[key] =_move_to_cuda(value)
                # return {
                #     key: _move_to_cuda(value)
                #     for key, value in maybe_tensor.items()
                # }
                return new_sample
            elif isinstance(maybe_tensor, nltkTree):
                return maybe_tensor
            elif isinstance(maybe_tensor, list):
                return [_move_to_cuda(x) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_cuda(sample)

    def _set_seed(self):
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.args.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)




def main(args, init_distributed=False):
    import_user_module(args)

    if args.max_tokens is None:
        args.max_tokens = 6000
    print(args)

    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load dataset splits
    valid_subsets = args.valid_subset.split(',')
    load_dataset_splits(task, ['train'] + valid_subsets)

    # Initialize distributed training (after data loading)
    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Make a dummy batch to (i) warm the caching allocator and (ii) as a
    # placeholder DistributedDataParallel when there's an uneven number of
    # batches per worker.
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        model.max_positions(),
    )
    dummy_batch = task.dataset('train').get_dummy_batch(args.max_tokens, max_positions)
    oom_batch = task.dataset('train').get_dummy_batch(1, max_positions)

    # Build trainer
    if args.model_parallel:
        print(f'| Model Parallel training  -> ModelParallel trainer')
        gpu_idx = list(range(torch.cuda.device_count()))
        trainer = ModelParallelTrainer(
            args, task, model, criterion, dummy_batch, oom_batch, gpu_idx=gpu_idx)
    else:
        trainer = FiTrainer(args, task, model, criterion, dummy_batch, oom_batch)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Initialize dataloader
    epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(args.train_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )

    # Load the latest checkpoint if one is available
    if not load_checkpoint(args, trainer, epoch_itr):
        trainer.dummy_train_step([dummy_batch])

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


class SafeAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        try:
            self.avg = self.sum / self.count
        except ZeroDivisionError as e:
            self.avg = 0


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    tune_epoch = getattr(args, 'tune_epoch', None)
    args.attempt_tune = True
    if tune_epoch is not None and epoch_itr.epoch > tune_epoch and args.attempt_tune:
        try:
            trainer.model.encoder.embed_tokens.turn_finetune(on=True)
        except Exception as e:
            print(f'Unable to turn finetuning on, tune_epoch={tune_epoch}')
            args.attempt_tune = False

    args.attempt_lroot_epoch = True
    # tune_epoch = getattr(args, 'tune_epoch', None)
    try:
        trainer.criterion.set_current_epoch(epoch_itr.epoch)
    except Exception as e:
        # print(f'Unable to set_current_epoch')
        args.attempt_lroot_epoch = False

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    acc_meter = collections.defaultdict(lambda: SafeAverageMeter())

    first_valid = args.valid_subset.split(',')[0]
    max_update = args.max_update or math.inf

    current_time = time.time()
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):

        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # if "target" in samples and

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in [
                'loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size',
                'bin_acc_sum',
                # 'bin_target_mean', 'target_mean',
                'nn_nsents'
            ]:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)

            stats[k] = extra_meters[k].avg

        if "bin_acc_sum" in log_output:
            assert 'nn_nsents' in log_output
            acc_meter['bin_acc'].update(log_output['bin_acc_sum'], log_output['nn_nsents'])

        if "bin_acc_avg" in log_output:
            assert 'nn_nsents' in log_output
            acc_meter['bin_acc_av'].update(log_output['bin_acc_avg'], log_output['nn_nsents'])

        if "acc" in log_output:
            acc_meter['all_acc'].update(log_output['acc'], log_output['nsentences'])

        if "target_mean" in log_output:
            acc_meter['all_target'].update(log_output['target_mean'], log_output['nsentences'])

        if "bin_target_mean" in log_output:
            acc_meter['all_bin_target'].update(log_output['bin_target_mean'], log_output['nn_nsents'])

        for k, v in acc_meter.items():
            stats[k] = v.avg

        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if args.save_interval_updates > 0 and num_updates % args.save_interval_updates == 0 and num_updates > 0:
            valid_losses = validate(args, trainer, task, epoch_itr, [first_valid])
            save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    end_time = time.time()
    laps = end_time - current_time
    print(f'| Epoch time: {laps}')

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    for k, v in acc_meter.items():
        stats[k] = v.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        kwargs = {}
        if '-bin' in subset:
            kwargs['filter_class_index'] = 2
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            # for SST-5-2
            # filter_class_index=2 if '-bin' in subset else None
            **kwargs
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        acc_meter = collections.defaultdict(lambda: SafeAverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in [
                    'loss', 'nll_loss', 'ntokens',
                    'nsentences', 'sample_size',
                    # 'target_mean', 'bin_target_mean',
                    'acc', 'bin_acc_sum'
                ]:
                    continue
                extra_meters[k].update(v)

            if "bin_acc_sum" in log_output:
                assert 'nn_nsents' in log_output
                acc_meter['bin_acc'].update(log_output['bin_acc_sum'], log_output['nn_nsents'])

            if "bin_acc_avg" in log_output:
                assert 'nn_nsents' in log_output
                acc_meter['bin_acc_av'].update(log_output['bin_acc_avg'], log_output['nn_nsents'])

            if "acc" in log_output:
                acc_meter['all_acc'].update(log_output['acc'], log_output['nsentences'])

            if "target_mean" in log_output:
                acc_meter['all_target'].update(log_output['target_mean'], log_output['nsentences'])

            if "bin_target_mean" in log_output:
                acc_meter['all_bin_target'].update(log_output['bin_target_mean'], log_output['nn_nsents'])

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg

        for k, v in acc_meter.items():
            stats[k] = v.avg

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats['loss'].avg)
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(save_checkpoint, 'best'):
        stats['best_loss'] = min(save_checkpoint.best, stats['loss'].avg)
    return stats


def get_perplexity(loss):
    try:
        return '{:.2f}'.format(math.pow(2, loss))
    except OverflowError:
        return float('inf')


def save_checkpoint(args, trainer, epoch_itr, val_loss):
    if args.no_save or not distributed_utils.is_master(args):
        return
    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
            end_of_epoch and not args.no_epoch_checkpoints and
            epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
            not end_of_epoch and args.save_interval_updates > 0 and
            updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
            val_loss is not None and
            (not hasattr(save_checkpoint, 'best') or val_loss < save_checkpoint.best)
    )
    checkpoint_conds['checkpoint_last.pt'] = True  # keep this last so that it's a symlink

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = min(val_loss, prev_best)
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.save_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        for cp in checkpoints:
            trainer.save_checkpoint(cp, extra_state)

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint_\d+_(\d+)\.pt')
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = utils.checkpoint_paths(args.save_dir, pattern=r'checkpoint(\d+)\.pt')
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer, epoch_itr):
    """Load a checkpoint and replay dataloader to match."""
    os.makedirs(args.save_dir, exist_ok=True)
    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.save_dir, args.restore_file)
    if os.path.isfile(checkpoint_path):
        extra_state = trainer.load_checkpoint(checkpoint_path, args.reset_optimizer, args.reset_lr_scheduler,
                                              eval(args.optimizer_overrides))
        if extra_state is not None:
            # replay train iterator to match checkpoint
            epoch_itr.load_state_dict(extra_state['train_iterator'])

            print('| loaded checkpoint {} (epoch {} @ {} updates)'.format(
                checkpoint_path, epoch_itr.epoch, trainer.get_num_updates()))

            trainer.lr_step(epoch_itr.epoch)
            trainer.lr_step_update(trainer.get_num_updates())
            if 'best' in extra_state:
                save_checkpoint.best = extra_state['best']
        return True
    else:
        print('| no existing checkpoint found {}'.format(checkpoint_path))
    return False


def load_dataset_splits(task, splits):
    for split in splits:
        if split == 'train':
            task.load_dataset(split, combine=True)
        else:
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')
                try:
                    task.load_dataset(split_k, combine=False)
                except FileNotFoundError as e:
                    if k > 0:
                        break
                    raise e


def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--model_parallel', action='store_true')

    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
