import copy
import typing

import numpy as np
import torch
import torch.utils.checkpoint

import revlib


def allclose(rtol: float = 1.e-5, atol: float = 1.e-8):
    def _fn(a: typing.Union[typing.List[typing.Union[int, float]], int, float],
            b: typing.Union[typing.List[typing.Union[int, float]], int, float]):
        return np.allclose(a, b, rtol, atol)

    return _fn


class BaseTest:
    def __init__(self):
        self.input_channels = 3
        self.inp = torch.randn((1, self.input_channels, 224, 224)).cuda()
        self.classes = 5
        self.dropout = 0.2
        self.conv_kernel = 3

        self.channels = 16
        self.channel_multiplier = 2
        self.classes = 1000

    def conv(self, in_channels, out_channels):
        return torch.nn.Conv2d(in_channels, out_channels, (self.conv_kernel,) * 2, padding=self.conv_kernel // 2)

    def block_conv(self, in_channels, out_channels):
        return torch.nn.Sequential(self.conv(in_channels, out_channels),
                                   torch.nn.Dropout(self.dropout),
                                   torch.nn.BatchNorm2d(out_channels, track_running_stats=False),
                                   torch.nn.ReLU())

    def block(self):
        return torch.nn.Sequential(self.block_conv(self.channels, self.channels * self.channel_multiplier),
                                   self.block_conv(self.channels * self.channel_multiplier, self.channels),
                                   torch.nn.Conv2d(self.channels, self.channels, (3, 3), padding=1))

    def rev_input(self):
        return self.conv(self.input_channels, 2 * self.channels)

    def rev_output(self):
        return self.conv(2 * self.channels, self.classes)

    def seq_input(self):
        return self.conv(self.input_channels, self.channels)

    def seq_output(self):
        return self.conv(self.channels, self.classes)

    def run(self, mod: torch.nn.Module):
        pass

    def compare(self, inp0: typing.Any, inp1: typing.Any):
        pass

    def revnet(self, blocks, memory_mode: revlib.MemoryModes):
        return revlib.ReversibleSequential(*blocks, memory_mode=memory_mode)

    def rng_run(self, mod: torch.nn.Module, cpu_state: torch.Tensor,
                cuda_state: typing.Tuple[typing.List[int], typing.List[torch.Tensor]]):
        torch.cuda.empty_cache()
        torch.set_rng_state(cpu_state)
        torch.utils.checkpoint.set_device_states(*cuda_state)
        return self.run(copy.deepcopy(mod))

    def run_and_compare(self, *modules: torch.nn.Module, comparison: typing.Callable):
        modules = [mod.cuda() for mod in modules]
        rng_state = torch.get_rng_state()
        cuda_state = torch.utils.checkpoint.get_device_states(self.inp)
        assert comparison(*(self.rng_run(mod, rng_state, cuda_state) for mod in modules))

    def __call__(self, *modules: torch.nn.Module, comparison: typing.Callable):
        pass


class RevTest(BaseTest):
    def __call__(self, *modules: torch.nn.Module, comparison: typing.Callable):
        inp_mod = self.rev_input()
        out_mod = self.rev_output()

        self.run_and_compare(*(torch.nn.Sequential(inp_mod, mod, out_mod) for mod in modules), comparison=comparison)
