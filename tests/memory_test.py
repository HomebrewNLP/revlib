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


def run(model: torch.nn.Module):
    _unused = model(inp)
    return torch.cuda.memory_allocated() * 2 ** -20


@pytest.mark.parametrize("depth", [16, 64])
def revnet_uses_less_test(depth: int):
    blocks = [block() for _ in range(depth)]
    rev = nn.Sequential(conv(3, 2 * channels), revlib.ReversibleSequential(*blocks), conv(2 * channels, classes)).cuda()
    seq = nn.Sequential(conv(3, channels), *blocks, conv(channels, classes)).cuda()
    assert run(rev) < run(seq)


@pytest.mark.parametrize("depth", [16, 64])
@pytest.mark.parametrize("factor", [2, 4, 32])
def no_more_memory_with_depth_test(depth: int, factor: int):
    blocks = [block() for _ in range(depth)]
    inp_mod = conv(3, 2 * channels)
    out_mod = conv(2 * channels, classes)
    rev0 = nn.Sequential(inp_mod, revlib.ReversibleSequential(*blocks), out_mod).cuda()
    rev1 = nn.Sequential(inp_mod, revlib.ReversibleSequential(*(blocks * factor)), out_mod).cuda()
    assert np.isclose(run(rev0), run(rev1), rtol=0.1)
