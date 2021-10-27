import pytest
import torch
from torch import nn

import revlib
from backend import RevTest, allclose


class MemTest(RevTest):
    def run(self, mod: torch.nn.Module):
        _unused = mod(self.inp)
        return torch.cuda.memory_allocated() * 2 ** -20


base = MemTest()


@pytest.mark.parametrize("depth", [16, 64])
@pytest.mark.parametrize("memory_mode", [revlib.MemoryModes.checkpoint, revlib.MemoryModes.autograd_graph,
                                         revlib.MemoryModes.autograd_function])
def revnet_uses_less_test(depth: int, memory_mode: revlib.MemoryModes):
    blocks = [base.block()] * depth
    rev = nn.Sequential(base.rev_input(), revlib.ReversibleSequential(*blocks, memory_mode=memory_mode),
                        base.rev_output()).cuda()
    seq = nn.Sequential(base.seq_input(), *blocks, base.seq_output()).cuda()
    base.run_and_compare(rev, seq, comparison=float.__lt__)


@pytest.mark.parametrize("depth", [8, 32])
@pytest.mark.parametrize("baseline", [revlib.MemoryModes.no_savings, revlib.MemoryModes.checkpoint])
@pytest.mark.parametrize("reversible", [revlib.MemoryModes.autograd_graph, revlib.MemoryModes.autograd_function])
def revnet_uses_more_without_savings_test(depth: int, baseline: revlib.MemoryModes, reversible: revlib.MemoryModes):
    blocks = [base.block()] * depth
    base(base.revnet(blocks, baseline), base.revnet(blocks, reversible), comparison=float.__gt__)


@pytest.mark.parametrize("depth", [8, 32])
@pytest.mark.parametrize("factor", [4, 16, 64])
@pytest.mark.parametrize("channels", [64])
@pytest.mark.parametrize("memory_mode", [revlib.MemoryModes.autograd_graph, revlib.MemoryModes.autograd_function])
def no_more_memory_with_depth_test(depth: int, factor: int, channels: int, memory_mode: revlib.MemoryModes):
    test = MemTest()
    test.channels = channels
    blocks = [test.block() for _ in range(depth)]
    test(base.revnet(blocks, memory_mode), base.revnet(blocks * factor, memory_mode),
         comparison=lambda x, y: allclose(0.1))
