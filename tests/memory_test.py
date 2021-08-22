import copy
import typing

import numpy as np
import pytest
import torch
from torch import nn

import revlib

channels = 16
channel_multiplier = 2
classes = 1000


def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)


def block_conv(in_channels, out_channels):
    return nn.Sequential(conv(in_channels, out_channels),
                         nn.Dropout(0.2),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


def block():
    return nn.Sequential(block_conv(channels, channels * channel_multiplier),
                         block_conv(channels * channel_multiplier, channels),
                         nn.Conv2d(channels, channels, (3, 3), padding=1))


inp = torch.randn((1, 3, 224, 224)).cuda()


def run(model: torch.nn.Module) -> float:
    _unused = model(inp)
    return torch.cuda.memory_allocated() * 2 ** -20


@pytest.mark.parametrize("depth", [16, 64])
def revnet_uses_less_test(depth: int):
    blocks = [block()] * depth
    rev = nn.Sequential(conv(3, 2 * channels), revlib.ReversibleSequential(*blocks), conv(2 * channels, classes)).cuda()
    seq = nn.Sequential(conv(3, channels), *blocks, conv(channels, classes)).cuda()
    assert run(rev) < run(seq)


def double_rev(stem0, stem1) -> typing.Tuple[float, float]:
    inp_mod = conv(3, 2 * channels)
    out_mod = conv(2 * channels, classes)
    rev0 = nn.Sequential(inp_mod, stem0, out_mod).cuda()
    rev1 = nn.Sequential(inp_mod, stem1, out_mod).cuda()
    return run(copy.deepcopy(rev0)), run(copy.deepcopy(rev1))


@pytest.mark.parametrize("depth", [8, 32])
def revnet_uses_more_without_savings_test(depth: int):
    blocks = [block()] * depth
    assert float.__gt__(*double_rev(revlib.ReversibleSequential(*blocks, memory_savings=False),
                                    revlib.ReversibleSequential(*blocks, memory_savings=True)))


@pytest.mark.parametrize("depth", [1, 4])
@pytest.mark.parametrize("factor", [4, 16, 64])
def no_more_memory_with_depth_test(depth: int, factor: int):
    blocks = [block() for _ in range(depth)]
    assert np.isclose(*double_rev(revlib.ReversibleSequential(*blocks),
                                  revlib.ReversibleSequential(*(blocks * factor))),
                      rtol=0.05)
