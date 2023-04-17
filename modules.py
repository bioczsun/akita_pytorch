import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.nn.functional as F

from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast


# from torchsummary import summary
from sklearn.metrics import r2_score, precision_score, f1_score

# from ray import tune

import json
import itertools
from itertools import groupby
import gzip
from io import BytesIO
from time import time

import matplotlib.pyplot as plt

import pyBigWig
from scipy.sparse import csc_matrix
import math
from copy import deepcopy


class StochasticReverseComplement(nn.Module):
    """Stochastically reverse complement a one hot encoded DNA sequence."""
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot, training=True):
        if training:
            rc_seq_1hot = seq_1hot[:, :, torch.tensor([3, 2, 1, 0], dtype=torch.long)]
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[1])
            reverse_bool = torch.rand(1) > 0.5
            src_seq_1hot = torch.where(reverse_bool, rc_seq_1hot, seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, torch.tensor(False)


class StochasticShift(nn.Module):
    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)
        else:
            self.augment_shifts = torch.arange(0, self.shift_max + 1)
        self.pad = pad

    def shift_sequence(self, seq_1hot, shift):
        # Shifts a sequence along the second axis
        if shift > 0:
            seq_1hot_padded = F.pad(seq_1hot, (0, 0, shift, 0), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, :-shift, :]
        else:
            seq_1hot_padded = F.pad(seq_1hot, (0, 0, 0, -shift), mode=self.pad)
            shifted_seq_1hot = seq_1hot_padded[:, -shift:, :]
        return shifted_seq_1hot

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.randint(len(self.augment_shifts), size=(1,))
            shift = self.augment_shifts[shift_i]
            sseq_1hot = torch.where(shift != 0,
                                    self.shift_sequence(seq_1hot, shift),
                                    seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot

    def extra_repr(self):
        return 'shift_max={}, symmetric={}, pad={}'.format(
            self.shift_max, self.symmetric, self.pad)

class OneToTwo(nn.Module):
    def __init__(self, operation='mean'):
        super(OneToTwo, self).__init__()
        self.operation = operation.lower()
        valid_operations = ['concat', 'mean', 'max', 'multiply', 'multiply1']
        assert self.operation in valid_operations

    def forward(self, oned):
        _, features,seq_len = oned.shape

        twod1 = oned.repeat(1, 1,seq_len)
        twod1 = twod1.view(-1, features, seq_len, seq_len)
        twod2 = torch.transpose(twod1, 1, 1)

        if self.operation == 'concat':
            twod = torch.cat([twod1, twod2], dim=-1)
        elif self.operation == 'multiply':
            twod = twod1 * twod2
        elif self.operation == 'multiply1':
            twod = (twod1 + 1) * (twod2 + 1) - 1
        else:
            twod1 = twod1.unsqueeze(-1)
            twod2 = twod2.unsqueeze(-1)
            twod = torch.cat([twod1, twod2], dim=-1)

            if self.operation == 'mean':
                twod = twod.mean(dim=-1)
            elif self.operation == 'max':
                twod = twod.max(dim=-1)[0]

        return twod




class ConcatDist2D(nn.Module):
    ''' Concatenate the pairwise distance to 2d feature matrix.'''

    def __init__(self):
        super(ConcatDist2D, self).__init__()

    def forward(self, inputs):
        batch_size, seq_len, features = inputs.shape[0],inputs.shape[3],inputs.shape[1]

        ## concat 2D distance ##
        pos = torch.arange(seq_len).unsqueeze(0).repeat(seq_len, 1)
        matrix_repr1 = pos
        matrix_repr2 = pos.t()
        dist = torch.abs(matrix_repr1 - matrix_repr2)
        dist = dist.float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, 1, 1)
        dist = torch.transpose(dist, 1, -1)

        return torch.cat([inputs, dist.cuda()], dim=1)


class Symmetrize2D(nn.Module):
    def __init__(self):
        super(Symmetrize2D, self).__init__()

    def forward(self, x):
        x_t = torch.transpose(x, 2,3)
        x_sym = (x + x_t) / 2
        return x_sym
class DilatedResidual2D(nn.Module):
    def __init__(self, in_channels, kernel_size, rate_mult, repeat, dropout):
        super(DilatedResidual2D, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2)

        # Define dilations
        dilations = [1]
        for i in range(1, repeat):
            dilations.append(int(i*rate_mult))

        # Define residual blocks
        self.res_blocks = nn.ModuleList()
        for dilation in dilations:
            self.res_blocks.append(nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, padding=dilation))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Residual block
        for block in self.res_blocks:
            res = out
            out = F.relu(out)
            out = self.dropout(out)

            out = block(out)
            out = F.relu(out)
            out = self.dropout(out)

            out = out + res

        return out

class Cropping2D(nn.Module):
    def __init__(self, cropping):
        super(Cropping2D, self).__init__()
        self.cropping = cropping

    def forward(self, inputs):
        _, _, h, w = inputs.size()
        cropped = inputs[:, :, self.cropping:h-self.cropping, self.cropping:w-self.cropping]
        return cropped


class UpperTri(nn.Module):
    def __init__(self, diagonal_offset=2):
        super(UpperTri, self).__init__()
        self.diagonal_offset = diagonal_offset

    def forward(self, inputs):
        seq_len = inputs.shape[2]
        output_dim = inputs.shape[1]

        triu_tup = np.triu_indices(seq_len, self.diagonal_offset)
        triu_index = list(triu_tup[0]+ seq_len*triu_tup[1])
        unroll_repr = inputs.reshape(-1, output_dim, seq_len**2)
        return torch.index_select(unroll_repr, 2, torch.tensor(triu_index))

    def extra_repr(self):
        return 'diagonal_offset={}'.format(self.diagonal_offset)


class Final(nn.Module):
    def __init__(self, l2_scale=0, l1_scale=0, **kwargs):
        super(Final, self).__init__()
        # self.flatten = nn.Flatten()
        self.l2_scale = l2_scale
        self.l1_scale = l1_scale
        self.dense = nn.Linear(in_features=12,out_features=1,bias=False)
    def forward(self,x):
        # x = self.flatten(x)
        # print(x.size())
        x = self.dense(x)
        # regularize
        if self.l2_scale > 0:
            x = F.normalize(x, p=2, dim=-1)
        if self.l1_scale > 0:
            x = F.normalize(x, p=1, dim=-1)

        return x

# def final(inputs, units, activation='linear', flatten=False,
#           kernel_initializer='he_normal', l2_scale=0, l1_scale=0, **kwargs):
#     """Final simple transformation before comparison to targets.
#     Args:
#         inputs:         [batch_size, seq_length, features] input sequence
#         units:          Dense units
#         activation:     relu/gelu/etc
#         flatten:        Flatten positional axis.
#         l2_scale:       L2 regularization weight.
#         l1_scale:       L1 regularization weight.
#     Returns:
#         [batch_size, seq_length(?), units] output sequence
#     """
#     current = inputs
#
#     # flatten
#     if flatten:
#         batch_size, seq_len, seq_depth = current.size()
#         current = current.view(batch_size, 1, seq_len * seq_depth)
#
#     # dense
#     current = nn.Linear(
#         in_features=current.size(-1),
#         out_features=units,
#         bias=True
#     )(current)
#     if activation == 'relu':
#         current = F.relu(current)
#     elif activation == 'gelu':
#         current = F.gelu(current)
#     elif activation == 'sigmoid':
#         current = torch.sigmoid(current)
#     elif activation == 'tanh':
#         current = torch.tanh(current)
#
#     # regularize
#     if l2_scale > 0:
#         current = F.normalize(current,p=2, dim=-1)
#     if l1_scale > 0:
#         current = F.normalize(current,p=1, dim=-1)
#
#     return current




class DilatedResidual1D(nn.Module):
    def __init__(self, filters, rate_mult, repeat, dropout):
        super(DilatedResidual1D, self).__init__()

        layers = []
        for i in range(repeat):
            dilation_rate = 2 ** (i % 10)
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        filters,
                        filters,
                        kernel_size=3,
                        dilation=dilation_rate,
                        padding=dilation_rate
                    ),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )

        self.layers = nn.Sequential(*layers)
        self.residual_proj = nn.Conv1d(filters, filters, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.layers(x)
        out += self.residual_proj(residual)
        out = self.dropout(out)
        out = self.activation(out)
        return out

class PearsonR(nn.Module):
    def __init__(self, num_targets, summarize=True):
        super(PearsonR, self).__init__()
        self.summarize = summarize
        self.shape = (num_targets,)
        self.count = nn.Parameter(torch.zeros(self.shape))

        self.product = nn.Parameter(torch.zeros(self.shape))
        self.true_sum = nn.Parameter(torch.zeros(self.shape))
        self.true_sumsq = nn.Parameter(torch.zeros(self.shape))
        self.pred_sum = nn.Parameter(torch.zeros(self.shape))
        self.pred_sumsq = nn.Parameter(torch.zeros(self.shape))

    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()

        if len(y_true.shape) == 2:
            reduce_axes = 0
        else:
            reduce_axes = [0,1]

        product = torch.sum(torch.mul(y_true, y_pred), dim=reduce_axes)
        self.product.data.add_(product)

        true_sum = torch.sum(y_true, dim=reduce_axes)
        self.true_sum.data.add_(true_sum)

        true_sumsq = torch.sum(torch.pow(y_true, 2), dim=reduce_axes)
        self.true_sumsq.data.add_(true_sumsq)

        pred_sum = torch.sum(y_pred, dim=reduce_axes)
        self.pred_sum.data.add_(pred_sum)

        pred_sumsq = torch.sum(torch.pow(y_pred, 2), dim=reduce_axes)
        self.pred_sumsq.data.add_(pred_sumsq)

        count = torch.ones_like(y_true)
        count = torch.sum(count, dim=reduce_axes)
        self.count.data.add_(count)

    def result(self):
        true_mean = torch.div(self.true_sum, self.count)
        true_mean2 = torch.pow(true_mean, 2)
        pred_mean = torch.div(self.pred_sum, self.count)
        pred_mean2 = torch.pow(pred_mean, 2)

        term1 = self.product
        term2 = -torch.mul(true_mean, self.pred_sum)
        term3 = -torch.mul(pred_mean, self.true_sum)
        term4 = torch.mul(self.count, torch.mul(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = self.true_sumsq - torch.mul(self.count, true_mean2)
        pred_var = self.pred_sumsq - torch.mul(self.count, pred_mean2)
        pred_var = torch.where(pred_var > 1e-12, pred_var, torch.full_like(pred_var, float('inf')))

        tp_var = torch.mul(torch.sqrt(true_var), torch.sqrt(pred_var))
        correlation = torch.div(covariance, tp_var)

        if self.summarize:
            return torch.mean(correlation)
        else:
            return correlation

    def reset_state(self):
        self.product.data.fill_(0)
        self.true_sum.data.fill_(0)
        self.true_sumsq.data.fill_(0)
        self.pred_sum.data.fill_(0)
        self.pred_sumsq.data.fill_(0)
        self.count.data.fill_(0)

