import typing

import torch.utils.checkpoint

from revlib.core import ReversibleSequential, MemoryModes, SingleBranchReversibleModule, take_0th_tensor, MergeCalls


class MomentumNetSide(torch.nn.Module):
    def __init__(self, beta: float):
        super(MomentumNetSide, self).__init__()
        self.beta = beta

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return inp * self.beta


class MomentumNetStem(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module, beta: float):
        super(MomentumNetStem, self).__init__()
        self.wrapped_module = wrapped_module
        self.beta = beta

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.wrapped_module(inp * self.beta, *args, **kwargs)


class ResidualToPlain(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module):
        super(ResidualToPlain, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        out = take_0th_tensor(self.wrapped_module(inp, *args, **kwargs))
        if isinstance(out, torch.Tensor):
            return out - inp
        return [out[0] - inp] + out[1]


def momentum_net(*modules, split_dim=1,
                 coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 memory_mode: MemoryModes = MemoryModes.autograd_function,
                 target_device: str = "",
                 beta: float = 0.9) -> ReversibleSequential:
    momentum_modules = []
    for idx, mod in enumerate(modules):
        momentum_modules.append(MomentumNetStem(mod, beta ** idx))
        momentum_modules.append(MomentumNetSide((1 - beta) / beta ** (idx + 1)))
    return ReversibleSequential(*momentum_modules, split_dim=split_dim, coupling_forward=coupling_forward,
                                coupling_inverse=coupling_inverse, memory_mode=memory_mode, target_device=target_device)


def residual_to_plain(*modules: torch.nn.Module) -> typing.List[ResidualToPlain]:
    return [ResidualToPlain(mod) for mod in modules]


def maybe_residual_to_plain(module: typing.Union[typing.List[torch.nn.Module], torch.nn.Sequential,
                                                 torch.nn.ModuleList],
                            residual: bool = False) -> typing.List[torch.nn.Module]:
    modules = list(module)
    if residual:
        modules = residual_to_plain(*modules)
    return modules


def sequential_to_revnet(module: torch.nn.Sequential,
                         split_dim=1,
                         coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                         coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                         memory_mode: MemoryModes = MemoryModes.autograd_function,
                         target_device: str = "",
                         residual: bool = False) -> ReversibleSequential:
    return ReversibleSequential(*maybe_residual_to_plain(module, residual), split_dim=split_dim,
                                coupling_forward=coupling_forward, coupling_inverse=coupling_inverse,
                                memory_mode=memory_mode, target_device=target_device)


def sequential_to_momentum_net(module: torch.nn.Sequential,
                               split_dim=1,
                               coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                               coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                               memory_mode: MemoryModes = MemoryModes.autograd_function,
                               target_device: str = "",
                               residual: bool = False,
                               beta: float = 0.9) -> ReversibleSequential:
    return momentum_net(*maybe_residual_to_plain(module, residual), split_dim=split_dim,
                        coupling_forward=coupling_forward, coupling_inverse=coupling_inverse, memory_mode=memory_mode,
                        target_device=target_device, beta=beta)


def module_list_to_momentum_net(module: torch.nn.ModuleList,
                                coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                                coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                                memory_mode: MemoryModes = MemoryModes.autograd_function,
                                target_device: str = "",
                                residual: bool = False,
                                beta: float = 0.9) -> torch.nn.ModuleList:
    net = momentum_net(*maybe_residual_to_plain(module, residual), split_dim=0, coupling_forward=coupling_forward,
                       coupling_inverse=coupling_inverse, memory_mode=memory_mode, target_device=target_device,
                       beta=beta)
    secondary_branch_buffer = []
    modules = [SingleBranchReversibleModule(secondary_branch_buffer, wrapped_module=mod.wrapped_module,
                                            coupling_forward=mod.coupling_forward,
                                            coupling_inverse=mod.coupling_inverse,
                                            memory_savings=mod.memory_savings, target_device=mod.target_device,
                                            cache=mod.cache, first=idx == 0)
               for idx, mod in enumerate(net.stem)]
    out_modules = [MergeCalls(modules[i], modules[i + 1], collate_fn=lambda y, x: [y] + x[0][1:])
                   for i in range(0, len(modules) - 2, 2)]
    out_modules.append(MergeCalls(modules[-2], modules[-1], collate_fn=lambda _, x: x[0]))
    return torch.nn.ModuleList(out_modules)
