import numpy as np
import pytest
import torch
from torch import nn

import revlib
from backend import RevTest


class MemTest(RevTest):
    def run(self, mod: torch.nn.Module):
        _unused = mod(self.inp)
        return torch.cuda.memory_allocated() * 2 ** -20


base = MemTest()


@pytest.mark.parametrize("depth", [16, 64])
def revnet_uses_less_test(depth: int):
    blocks = [base.block()] * depth
    rev = nn.Sequential(base.rev_input(), revlib.ReversibleSequential(*blocks), base.rev_output()).cuda()
    seq = nn.Sequential(base.seq_input(), *blocks, base.seq_output()).cuda()
    base.run_and_compare(rev, seq, float.__lt__)


@pytest.mark.parametrize("depth", [8, 32])
def revnet_uses_more_without_savings_test(depth: int):
    blocks = [base.block()] * depth
    base(base.revnet(blocks, False), base.revnet(blocks, True), float.__gt__)


@pytest.mark.parametrize("depth", [8, 32])
@pytest.mark.parametrize("factor", [4, 16, 64])
@pytest.mark.parametrize("channels", [64])
def no_more_memory_with_depth_test(depth: int, factor: int, channels: int):
    test = MemTest()
    test.channels = channels
    blocks = [test.block() for _ in range(depth)]
    test(base.revnet(blocks), base.revnet(blocks * factor), lambda x, y: np.isclose(x, y, rtol=0.1))
