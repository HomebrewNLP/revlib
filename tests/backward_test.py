import typing

import numpy as np
import pytest
import torch

from backend import RevTest


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
def same_gradients_without_memory_savings_test(depth: int, weight_sharing: bool):
    blocks = ([base.block()] * depth) if weight_sharing else [base.block() for _ in range(depth)]
    base(base.revnet(blocks, False), base.revnet(blocks, True), np.allclose)

