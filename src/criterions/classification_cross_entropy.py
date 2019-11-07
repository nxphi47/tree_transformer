import math

from fairseq import utils
import torch
import torch.nn.functional as F
import torch.nn as nn

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from copy import deepcopy
import numpy as np

def accuracy(logits, targets, reduce=True):
    preds = logits.argmax(-1).long()
    targets = targets.to(preds.device)
    acc = (preds == targets).float()
    if reduce:
        acc = acc.sum()
    return acc


# def tf_binary_hits(logits, labels):
#   softmax = tf.nn.softmax(logits)
#   binary_predictions = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
#   binary_labels = labels > 2
#   return tf.cast(tf.equal(binary_predictions, binary_labels), tf.float64)

def binary_sentiment_accuracy(logits, targets, reduce=True):
    # targets = targets.to(logits.device)
    softmax = F.softmax(logits, dim=-1)
    preds = ((softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])).long()
    acc = (preds == targets).float()
    if reduce:
        acc = acc.sum()
    return acc


# nn.L1Loss

# def weight_rec_loss(w, loss_f):
#     diff = torch.matmul(w, w) - w

@register_criterion('classification_cross_entropy')
class ClassificationCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        # self.eps = args.label_smoothing
        self.l1_loss = nn.L1Loss(reduce=True, reduction='mean')
        self.warning_add_rec_param_loss = self.args.add_rec_param_loss

    @staticmethod
    def add_args(parser):
        """
        [This is just there to conform the training script]
        Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report_binary', action='store_true', help='report_binary')
        parser.add_argument('--add_rec_param_loss', action='store_true', help='add_rec_param_loss')

    def maybe_add_rec_param_loss(self, loss, model):
        if not self.args.add_rec_param_loss:
            return loss
        try:
            rec_params = model.recusive_params
            assert len(rec_params) > 0
        except Exception as e:
            if self.warning_add_rec_param_loss:
                if 'rec_params' in locals():
                    print(f'WARNING!!!!!: rec_params-faile: {rec_params}')
                else:
                    print(f'WARNING!!!!!: Cannot retrieve rec_params: {e}')
                self.warning_add_rec_param_loss = False
            return loss

        rec_losses = [self.l1_loss(torch.matmul(x, x), x).type_as(loss) for x in rec_params]
        for l in rec_losses:
            loss += l
        return loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        # print(f'Criteron logits net_output: {net_output}')
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)

        # print(f'Criteron loss finish')
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        acc, target_sum = self.compute_accuracy(model, net_output, sample, reduce=reduce)
        # print(f'Criteron acc finish')

        # print(f'{loss.device}')
        # print(f'{loss.dtype}')
        loss_data = loss.data
        # print(f'Criteron loss_data finish, {loss_data.device}')
        # print(f'Criteron loss_data finish data, {loss.data}')
        _loss = utils.item(loss_data) if reduce else loss_data
        # print(f'Criteron _loss finish')
        _acc = utils.item(acc.data)
        # print(f'Criteron _acc finish')
        _target_sum = utils.item(target_sum.data)
        # print(f'Criteron _target_sum finish')

        logging_output = {
            'loss': _loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'acc': _acc,
            'target_sum': _target_sum,
            'sample_size': sample_size,
        }

        if self.args.report_binary:
            # print(f'Binary computed here')
            bin_acc, bin_target_sum, non_neutral_nseq = self.compute_accuracy(
                model, net_output, sample, reduce=reduce, report_binary=True)
            logging_output['nn_nsents'] = utils.item(non_neutral_nseq.data)
            logging_output['bin_acc'] = utils.item(bin_acc.data)
            logging_output['bin_target_sum'] = utils.item(bin_target_sum.data)

        # print(f'Criteron finish')
        return loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample, reduce=True, report_binary=False):
        net_output = net_output.view(-1, net_output.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        target = target.to(net_output.device)
        logits = net_output
        # if self.args.report_binary:
        #     acc = binary_sentiment_accuracy(net_output, target, reduce=reduce)
        # else:
        #     acc = accuracy(net_output, target, reduce=reduce)
        #
        if report_binary:
            non_neutral_mask = target.ne(2)
            non_neutral_target = target[non_neutral_mask]
            non_neutral_logits = logits[non_neutral_mask]
            assert non_neutral_target.size(0) == non_neutral_logits.size(0)
            non_neutral_nseq = non_neutral_mask.int().sum()

            binary_target = (non_neutral_target > 2).type_as(non_neutral_target)
            target_sum = binary_target.sum()
            acc = binary_sentiment_accuracy(non_neutral_logits, binary_target, reduce=reduce)
            return acc, target_sum, non_neutral_nseq
        else:
            acc = accuracy(net_output, target, reduce=reduce)
            target_sum = target.sum()
            return acc, target_sum

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        # print(f'target device -> [{lprobs.device}]: {target}')
        # print(f'target device -> [{target.device}]')
        # target = target.to(lprobs.device)
        # export lprobs [b, C]
        # export target [b]
        assert lprobs.size(0) == target.size(0), f'{lprobs.size()} != {target.size()}'
        # print(f'computing losss')
        # print(target)
        # print(lprobs)
        loss = F.nll_loss(lprobs, target, size_average=False, reduce=reduce)
        # print(f'Printing loss')
        # print(loss)
        # loss = self.maybe_add_rec_param_loss(loss, model)
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        acc_sum = sum(log.get('acc', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'acc': acc_sum / float(nsentences),
            # 'acc': acc_sum,
            'acc_sum': acc_sum,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if 'target_sum' in logging_outputs[0]:
            agg_output['target_mean'] = sum(log.get('target_sum', 0) for log in logging_outputs) / float(nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)

        if "bin_acc" in logging_outputs[0]:
            assert "nn_nsents" in logging_outputs[0]
            bin_acc_sum = sum(log.get('bin_acc', 0) for log in logging_outputs)
            bin_target_sum = sum(log.get('bin_target_sum', 0) for log in logging_outputs)
            nn_nsents = sum(log.get('nn_nsents', 0) for log in logging_outputs)
            agg_output['nn_nsents'] = nn_nsents
            agg_output['bin_acc_sum'] = bin_acc_sum
            agg_output['bin_acc_avg'] = bin_acc_sum / float(max(nn_nsents, 1))
            agg_output['bin_target_mean'] = bin_target_sum / float(max(nn_nsents, 1))
        return agg_output


@register_criterion('subtree_classification_label_smoothed_cross_entropy')
class SubtreeClassificationLabelSmoothedCrossEntropyCriterion(ClassificationCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = getattr(args, 'label_smoothing', 0.0)
        self.report_binary = getattr(args, 'report_binary', False)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report_binary', action='store_true', help='report_binary')
        parser.add_argument('--add_rec_param_loss', action='store_true', help='add_rec_param_loss')
        parser.add_argument('--reduce_avg', action='store_true', help='reduce_avg')
        parser.add_argument('--reduce_token', action='store_true', help='reduce_token')
        parser.add_argument('--reduce_sent', action='store_true', help='reduce_sent')

        parser.add_argument('--only_first_root', action='store_true', help='only_first_root')

    def get_target_from_sample_for_eval(self, sample, binary_target=False):
        target = sample['net_input']['src_labels']
        return target

    def get_target_from_sample_for_train(self, sample):
        target = sample['net_input']['src_labels']
        return target

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        # Encoder_out(T x B x m x C), in the model, convert to --> B x m x T x C

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        # sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        sample_size = sample['net_input']['src_labels'].size(0) if self.args.sentence_avg else sample['ntokens']

        acc, target_sum = self.compute_accuracy(model, net_output, sample, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            # 'nsentences': sample['target'].size(0),
            'nsentences': sample['net_input']['src_labels'].size(0),
            'acc': utils.item(acc.data),
            'target_sum': utils.item(target_sum.data),
            'sample_size': sample_size,
        }

        if self.args.report_binary:
            bin_acc, bin_target_sum, non_neutral_nseq = self.compute_accuracy(
                model, net_output, sample, reduce=reduce, report_binary=True)
            logging_output['nn_nsents'] = utils.item(non_neutral_nseq.data)
            logging_output['bin_acc'] = utils.item(bin_acc.data)
            logging_output['bin_target_sum'] = utils.item(bin_target_sum.data)

        return loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample, reduce=True, report_binary=False):
        logits = net_output[:, 0, 0]
        logits = logits.view(-1, logits.size(-1))
        target = self.get_target_from_sample_for_eval(sample)[:, 0, 0]
        nodes = sample['net_input']['src_tokens']

        if report_binary:
            non_neutral_mask = target.ne(2)
            non_neutral_target = target[non_neutral_mask]
            non_neutral_logits = logits[non_neutral_mask]
            assert non_neutral_target.size(0) == non_neutral_logits.size(0)
            non_neutral_nseq = non_neutral_mask.int().sum()

            binary_target = (non_neutral_target > 2).type_as(non_neutral_target)
            target_sum = binary_target.sum()
            acc = binary_sentiment_accuracy(non_neutral_logits, binary_target, reduce=reduce)
            return acc, target_sum, non_neutral_nseq
        else:
            if self.args.only_binary:
                assert target.ne(2).all(), f'target 2: {target}'
                binary_target = (target > 2).type_as(target)
                target_sum = binary_target.sum()
                acc = binary_sentiment_accuracy(logits, binary_target, reduce=reduce)
                return acc, target_sum
            else:
                acc = accuracy(logits, target, reduce=reduce)
                target_sum = target.sum()
                return acc, target_sum

    def compute_loss(self, model, net_output, sample, reduce=True):
        """

        :param model:
        :param net_output:          [b, m, t, c]
        :param sample:  -> nodes    [b, m, t], labels [b, m, t]
        :param reduce:
        :return:
        """
        if self.args.only_first_root:
            net_output = net_output[:, 0, 0]
            target = model.get_targets(sample, net_output)[:, 0, 0].view(-1)

            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            # export lprobs [b, C]
            # export target [b]
            assert lprobs.size(0) == target.size(0), f'{lprobs.size()} != {target.size()}'
            loss = F.nll_loss(lprobs, target, size_average=False, reduce=reduce)
            loss = self.maybe_add_rec_param_loss(loss, model)
            return loss, loss

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = self.get_target_from_sample_for_train(sample)
        batch_size = target.size(0)
        nodes = sample['net_input']['src_tokens']
        # both:         [b, m, t]
        target = target.view(-1, 1)
        nodes = nodes.view(-1, 1)

        non_pad_mask = nodes.ne(self.padding_idx)
        assert lprobs.size(0) == target.size(0), f'{lprobs.size()} != {target.size()}'
        assert lprobs.size(0) == nodes.size(0), f'{lprobs.size()} != {nodes.size()}'
        # print(f'lprob: {lprobs.size()} [{target.max()}][{nodes.max()}], {target}, {nodes}')

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            if self.args.reduce_avg or self.args.reduce_token:
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()
            elif self.args.reduce_sent:
                nll_loss = nll_loss.sum() / batch_size
                smooth_loss = smooth_loss.sum() / batch_size
            else:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        loss = self.maybe_add_rec_param_loss(loss, model)
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        acc_sum = sum(log.get('acc', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'acc': acc_sum / float(nsentences),
            # 'target_sum': target_sum / float(nsentences),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)

        if 'target_sum' in logging_outputs[0]:
            agg_output['target_mean'] = sum(log.get('target_sum', 0) for log in logging_outputs) / float(nsentences)
        
        if "bin_acc" in logging_outputs[0]:
            assert "nn_nsents" in logging_outputs[0]
            bin_acc_sum = sum(log.get('bin_acc', 0) for log in logging_outputs)
            bin_target_sum = sum(log.get('bin_target_sum', 0) for log in logging_outputs)
            nn_nsents = sum(log.get('nn_nsents', 0) for log in logging_outputs)
            agg_output['nn_nsents'] = nn_nsents
            agg_output['bin_acc_sum'] = bin_acc_sum
            agg_output['bin_acc_avg'] = bin_acc_sum / float(max(nn_nsents, 1))
            agg_output['bin_target_mean'] = bin_target_sum / float(max(nn_nsents, 1))

        return agg_output


@register_criterion('subtree2root_classification_label_smoothed_cross_entropy')
class Subtree2RootClassificationLabelSmoothedCrossEntropyCriterion(SubtreeClassificationLabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = getattr(args, 'label_smoothing', 0.0)
        self.report_binary = getattr(args, 'report_binary', False)
        self.lroot_epoch = getattr(args, 'lroot_epoch', -1)

        self.current_epoch = 0
        self.flip_switch = False

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
        if self.current_epoch > self.lroot_epoch and not self.flip_switch:
            print(f'lroot_epoch {self.lroot_epoch} reached, switch to root training')
            self.flip_switch = True

    @staticmethod
    def add_args(parser):
        SubtreeClassificationLabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--lroot_epoch", metavar="N", default=-1, type=int,
                           help="lroot_epoch")

    def compute_accuracy(self, model, net_output, sample, reduce=True, report_binary=False):
        logits = net_output[:, 0, 0]
        logits = logits.view(-1, logits.size(-1))
        target = self.get_target_from_sample_for_eval(sample)[:, 0, 0]
        nodes = sample['net_input']['src_tokens']

        if report_binary:
            non_neutral_mask = target.ne(2)
            non_neutral_target = target[non_neutral_mask]
            non_neutral_logits = logits[non_neutral_mask]
            assert non_neutral_target.size(0) == non_neutral_logits.size(0)
            non_neutral_nseq = non_neutral_mask.int().sum()

            binary_target = (non_neutral_target > 2).type_as(non_neutral_target)
            target_sum = binary_target.sum()
            acc = binary_sentiment_accuracy(non_neutral_logits, binary_target, reduce=reduce)
            return acc, target_sum, non_neutral_nseq
        else:
            if self.args.only_binary:
                assert target.ne(2).all(), f'target 2: {target}'
                binary_target = (target > 2).type_as(target)
                target_sum = binary_target.sum()
                acc = binary_sentiment_accuracy(logits, binary_target, reduce=reduce)
                return acc, target_sum
            else:
                acc = accuracy(logits, target, reduce=reduce)
                target_sum = target.sum()
                return acc, target_sum

    def compute_loss(self, model, net_output, sample, reduce=True):
        """

        :param model:
        :param net_output:          [b, m, t, c]
        :param sample:  -> nodes    [b, m, t], labels [b, m, t]
        :param reduce:
        :return:
        """
        target = self.get_target_from_sample_for_train(sample)
        nodes = sample['net_input']['src_tokens']

        if self.current_epoch > self.lroot_epoch:
            target = target[:, :1, :1]
            nodes = nodes[:, :1, :1]
            net_output = net_output[:, :1, :1]

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        # both:         [b, m, t]
        target = target.view(-1, 1)
        nodes = nodes.view(-1, 1)

        non_pad_mask = nodes.ne(self.padding_idx)
        assert lprobs.size(0) == target.size(0), f'{lprobs.size()} != {target.size()}'
        assert lprobs.size(0) == nodes.size(0), f'{lprobs.size()} != {nodes.size()}'
        # print(f'lprob: {lprobs.size()} [{target.max()}][{nodes.max()}], {target}, {nodes}')

        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            if self.args.reduce_avg:
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()
            else:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        loss = self.maybe_add_rec_param_loss(loss, model)
        return loss, nll_loss


@register_criterion('nstack_node_class_cross_entropy')
class NstackNodeClassCrossEntropyCriterion(ClassificationCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = getattr(args, 'label_smoothing', 0.0)
        self.report_binary = getattr(args, 'report_binary', False)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report_binary', action='store_true', help='report_binary')
        parser.add_argument('--add_rec_param_loss', action='store_true', help='add_rec_param_loss')
        parser.add_argument('--reduce_avg', action='store_true', help='reduce_avg')
        parser.add_argument('--reduce_token', action='store_true', help='reduce_token')
        parser.add_argument('--reduce_sent', action='store_true', help='reduce_sent')

        parser.add_argument('--only_first_root', action='store_true', help='only_first_root')

    def get_target_from_sample_for_eval(self, sample, binary_target=False):
        target = sample['net_input']['src_labels']
        return target

    def get_target_from_sample_for_train(self, sample):
        target = sample['net_input']['src_labels']
        return target

    def forward(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['net_input']['src_labels'].size(0) if self.args.sentence_avg else sample['ntokens']
        acc, target_sum = self.compute_accuracy(model, net_output, sample, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            # 'nsentences': sample['target'].size(0),
            'nsentences': sample['net_input']['src_labels'].size(0),
            'acc': utils.item(acc.data),
            'target_sum': utils.item(target_sum.data),
            'sample_size': sample_size,
        }

        if self.args.report_binary:
            bin_acc, bin_target_sum, non_neutral_nseq = self.compute_accuracy(
                model, net_output, sample, reduce=reduce, report_binary=True)
            logging_output['nn_nsents'] = utils.item(non_neutral_nseq.data)
            logging_output['bin_acc'] = utils.item(bin_acc.data)
            logging_output['bin_target_sum'] = utils.item(bin_target_sum.data)

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        """

        :param model:
        :param net_output:          [b, t, c]
        :param sample:  -> nodes    [b, t], labels [b, t]
        :param reduce:
        :return:
        """
        if self.args.only_first_root:
            net_output = net_output[:, -1]
            tgt_toks = model.get_targets(sample, net_output)[:, -1].view(-1)

            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            # export lprobs [b, C]
            # export target [b]
            assert lprobs.size(0) == tgt_toks.size(0), f'{lprobs.size()} != {tgt_toks.size()}'
            loss = F.nll_loss(lprobs, tgt_toks, size_average=False, reduce=reduce)
            loss = self.maybe_add_rec_param_loss(loss, model)
            return loss, loss

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        tgt_toks = self.get_target_from_sample_for_train(sample)
        batch_size = tgt_toks.size(0)
        src_toks = sample['net_input']['src_tokens']
        # both:         [b, m, t]
        tgt_toks = tgt_toks.view(-1, 1)
        src_toks = src_toks.view(-1, 1)

        non_pad_mask = src_toks.ne(self.padding_idx)
        assert lprobs.size(0) == tgt_toks.size(0), f'{lprobs.size()} != {tgt_toks.size()} - {src_toks.size()}'
        assert lprobs.size(0) == src_toks.size(0), f'{lprobs.size()} != {tgt_toks.size()} - {src_toks.size()}'

        nll_loss = -lprobs.gather(dim=-1, index=tgt_toks)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            if self.args.reduce_avg or self.args.reduce_token:
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()
            elif self.args.reduce_sent:
                nll_loss = nll_loss.sum() / batch_size
                smooth_loss = smooth_loss.sum() / batch_size
            else:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        loss = self.maybe_add_rec_param_loss(loss, model)
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample, reduce=True, report_binary=False):
        logits = net_output[:, -1]
        logits = logits.view(-1, logits.size(-1))
        target = self.get_target_from_sample_for_eval(sample)[:, -1]
        # nodes = sample['net_input']['src_tokens']

        if report_binary:
            non_neutral_mask = target.ne(2)
            non_neutral_target = target[non_neutral_mask]
            non_neutral_logits = logits[non_neutral_mask]
            assert non_neutral_target.size(0) == non_neutral_logits.size(0)
            non_neutral_nseq = non_neutral_mask.int().sum()

            binary_target = (non_neutral_target > 2).type_as(non_neutral_target)
            target_sum = binary_target.sum()
            acc = binary_sentiment_accuracy(non_neutral_logits, binary_target, reduce=reduce)
            return acc, target_sum, non_neutral_nseq
        else:
            if self.args.only_binary:
                assert target.ne(2).all(), f'target 2: {target}'
                binary_target = (target > 2).type_as(target)
                target_sum = binary_target.sum()
                acc = binary_sentiment_accuracy(logits, binary_target, reduce=reduce)
                return acc, target_sum
            else:
                acc = accuracy(logits, target, reduce=reduce)
                target_sum = target.sum()
                return acc, target_sum

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        acc_sum = sum(log.get('acc', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'acc': acc_sum / float(nsentences),
            # 'target_sum': target_sum / float(nsentences),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)

        if 'target_sum' in logging_outputs[0]:
            agg_output['target_mean'] = sum(log.get('target_sum', 0) for log in logging_outputs) / float(nsentences)

        if "bin_acc" in logging_outputs[0]:
            assert "nn_nsents" in logging_outputs[0]
            bin_acc_sum = sum(log.get('bin_acc', 0) for log in logging_outputs)
            bin_target_sum = sum(log.get('bin_target_sum', 0) for log in logging_outputs)
            nn_nsents = sum(log.get('nn_nsents', 0) for log in logging_outputs)
            agg_output['nn_nsents'] = nn_nsents
            agg_output['bin_acc_sum'] = bin_acc_sum
            agg_output['bin_acc_avg'] = bin_acc_sum / float(max(nn_nsents, 1))
            agg_output['bin_target_mean'] = bin_target_sum / float(max(nn_nsents, 1))

        return agg_output


@register_criterion('nstack_node_lm_class_cross_entropy')
class NstackNodeLMClassCrossEntropyCriterion(NstackNodeClassCrossEntropyCriterion):

    @staticmethod
    def add_args(parser):
        NstackNodeClassCrossEntropyCriterion.add_args(parser)
        parser.add_argument('--lm_temp', default=1.0, type=float, metavar='D', help='LM temparature')

    def get_nodes(self, sample):
        return sample['net_input']['src_node_nodes']

    def forward(self, model, sample, reduce=True):
        _net_output_ = model(**sample['net_input'])
        class_output, node_output = _net_output_

        class_loss, nll_loss_class = self.compute_loss(model, class_output, sample, reduce=reduce)
        lm_loss, nll_loss_lm = self.compute_lm_loss(model, node_output, sample, reduce=reduce)
        loss = class_loss + self.args.lm_temp * lm_loss
        # nll_loss = nll_loss_class + self.args.lm_temp * nll_loss_lm

        sample_size = sample['net_input']['src_labels'].size(0) if self.args.sentence_avg else sample['ntokens']
        acc, target_sum = self.compute_accuracy(model, class_output, sample, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            # 'nsentences': sample['target'].size(0),
            'nsentences': sample['net_input']['src_labels'].size(0),
            'acc': utils.item(acc.data),
            'target_sum': utils.item(target_sum.data),
            'sample_size': sample_size,
        }

        if self.args.report_binary:
            bin_acc, bin_target_sum, non_neutral_nseq = self.compute_accuracy(
                model, class_output, sample, reduce=reduce, report_binary=True)
            logging_output['nn_nsents'] = utils.item(non_neutral_nseq.data)
            logging_output['bin_acc'] = utils.item(bin_acc.data)
            logging_output['bin_target_sum'] = utils.item(bin_target_sum.data)

        return loss, sample_size, logging_output

    def compute_lm_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        # target = model.get_targets(sample, net_output).view(-1, 1)
        target =self.get_nodes(sample).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        return loss, nll_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        """

        :param model:
        :param net_output:          [b, t, c]
        :param sample:  -> nodes    [b, t], labels [b, t]
        :param reduce:
        :return:
        """
        if self.args.only_first_root:
            raise NotImplementedError

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        tgt_toks = self.get_target_from_sample_for_train(sample)
        batch_size = tgt_toks.size(0)
        src_toks = sample['net_input']['src_tokens']
        # both:         [b, m, t]
        tgt_toks = tgt_toks.view(-1, 1)
        src_toks = src_toks.view(-1, 1)

        non_pad_mask = src_toks.ne(self.padding_idx)
        assert lprobs.size(0) == tgt_toks.size(0), f'{lprobs.size()} != {tgt_toks.size()} - {src_toks.size()}'
        assert lprobs.size(0) == src_toks.size(0), f'{lprobs.size()} != {tgt_toks.size()} - {src_toks.size()}'

        nll_loss = -lprobs.gather(dim=-1, index=tgt_toks)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            if self.args.reduce_avg or self.args.reduce_token:
                nll_loss = nll_loss.mean()
                smooth_loss = smooth_loss.mean()
            elif self.args.reduce_sent:
                nll_loss = nll_loss.sum() / batch_size
                smooth_loss = smooth_loss.sum() / batch_size
            else:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        loss = self.maybe_add_rec_param_loss(loss, model)
        return loss, nll_loss




def pearson_example(predictions, labels):
    x = deepcopy(predictions)
    y = deepcopy(labels)
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    return torch.mean(torch.mul(x, y))


@register_criterion('nstack_relate_class_kldiv')
class NstackRelateKLCrossEntropyCriterion(ClassificationCrossEntropyCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.crit = nn.KLDivLoss(size_average=False)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        # target = model.get_targets(sample, net_output)
        tgt_score = sample['tgt_score']
        preds = self.make_predictions(net_output)
        assert preds.size() == tgt_score.size()
        # acc, target_sum = self.compute_accuracy(model, net_output, sample, reduce=reduce)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'preds': preds.detach().cpu().numpy(),
            'tgt_scores': tgt_score.detach().cpu().numpy(),
            'sample_size': sample_size,
        }

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        # lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output)
        tgt_score = sample['tgt_score']
        assert lprobs.size() == target.size(), f'{lprobs.size()}, {target.size()}, {tgt_score.size()}'
        # loss = self.crit(lprobs, target)
        loss = F.kl_div(lprobs, target, size_average=False, reduce=reduce)
        return loss, loss

    def make_predictions(self, net_output):
        # net_output = model(**sample['net_input'])
        nclass = net_output.size(-1)
        softmax = torch.softmax(net_output, -1)
        # softmax = softmax.float().cpu()
        indices = torch.arange(1, nclass + 1, dtype=softmax.dtype, device=softmax.device).view(1, nclass)
        # print(f"Pred: {softmax.size()}, {indices.size()}")
        preds = (softmax * indices).sum(-1)
        return preds

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        preds = np.concatenate([log.get('preds') for log in logging_outputs], 0)
        tgt_scores = np.concatenate([log.get('tgt_scores') for log in logging_outputs], 0)

        agg_output = {
            'loss': loss_sum / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,

            'preds': preds,
            'tgt_scores': tgt_scores,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens

        return agg_output



