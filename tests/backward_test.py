import typing

import pytest
import torch

import revlib
from backend import RevTest, allclose


class GradTest(RevTest):
    def __init__(self):
        super(GradTest, self).__init__()
        self.reductions: typing.List[typing.Callable] = [torch.mean, torch.std, torch.max, torch.min]

    def run(self, mod: torch.nn.Module):
        out = mod(self.inp)
        out.mean().backward()
        return [r(p.grad).item() for p in mod.parameters() for r in self.reductions]


base = GradTest()


@pytest.mark.parametrize("depth", [1, 8])
@pytest.mark.parametrize("weight_sharing", [True, False])
@pytest.mark.parametrize("memory_mode", [revlib.MemoryModes.checkpoint, revlib.MemoryModes.autograd_graph,
                                         revlib.MemoryModes.autograd_function])
def same_gradients_without_memory_savings_test(depth: int, weight_sharing: bool, memory_mode: revlib.MemoryModes):
    blocks = ([base.block()] * depth) if weight_sharing else [base.block() for _ in range(depth)]
    base(base.revnet(blocks, revlib.MemoryModes.no_savings), base.revnet(blocks, memory_mode),
         comparison=allclose(1e-3))
