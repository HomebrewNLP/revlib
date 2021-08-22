import copy
import typing

import torch

import revlib


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
                                   torch.nn.BatchNorm2d(out_channels),
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

    def revnet(self, blocks, memory_savings=True):
        return revlib.ReversibleSequential(*blocks, memory_savings=memory_savings)

    def run_and_compare(self, mod0: torch.nn.Module, mod1: torch.nn.Module, comparison: typing.Callable):
        mod0 = mod0.cuda()
        mod1 = mod1.cuda()
        assert comparison(self.run(copy.deepcopy(mod0)), self.run(copy.deepcopy(mod1)))

    def __call__(self, mod0: torch.nn.Module, mod1: torch.nn.Module, comparison: typing.Callable):
        pass


class RevTest(BaseTest):
    def __call__(self, mod0: torch.nn.Module, mod1: torch.nn.Module, comparison: typing.Callable):
        inp_mod = self.rev_input()
        out_mod = self.rev_output()
        self.run_and_compare(torch.nn.Sequential(inp_mod, mod0, out_mod), torch.nn.Sequential(inp_mod, mod1, out_mod),
                             comparison)
