import os
import secrets
import typing

import torch.utils.checkpoint
from bitsandbytes.functional import quantize_blockwise, dequantize_blockwise
from torch.utils._pytree import tree_map

from revlib.core import ReversibleSequential, MemoryModes, SingleBranchReversibleModule, split_tensor_list, MergeCalls


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
        out = split_tensor_list(self.wrapped_module(inp, *args, **kwargs))
        if isinstance(out, torch.Tensor):
            return out - inp
        return [out[0] - inp] + out[1]


def apply_tree(obj, fn: typing.Callable[[typing.Any], typing.Any]):
    if hasattr(obj, '__dict__'):
        obj.__dict__ = apply_tree(obj.__dict__, fn)
    if isinstance(obj, dict):
        return dict(zip(apply_tree(list(obj.keys()), fn), apply_tree(list(obj.values()), fn)))
    if isinstance(obj, (tuple, list)):
        return type(obj)([apply_tree(o, fn) for o in obj])
    return fn(obj)


def detached_additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor
                                       ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
    fn_out = split_tensor_list(fn_out)
    if isinstance(fn_out, torch.Tensor):
        return other_stream + fn_out
    return [other_stream + fn_out[0]] + apply_tree(fn_out[1], torch.detach)


def detached_additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor
                                       ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
    fn_out = split_tensor_list(fn_out)
    if isinstance(fn_out, torch.Tensor):
        return output - fn_out
    return [output - fn_out[0]] + apply_tree(fn_out[1], torch.detach)


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
    stem = list(net)[:-1]  # Drop last `MomentumNetSide`
    modules = [SingleBranchReversibleModule(secondary_branch_buffer, wrapped_module=mod.wrapped_module.wrapped_module,
                                            coupling_forward=mod.wrapped_module.coupling_forward,
                                            coupling_inverse=mod.wrapped_module.coupling_inverse,
                                            memory_savings=mod.memory_savings, target_device=mod.target_device,
                                            cache=mod.cache, first=idx == 0, last=idx == len(stem) - 1)
               for idx, mod in enumerate(stem)]
    out_modules = [MergeCalls(modules[i], modules[i + 1], collate_fn=lambda y, x: [y] + x[0][1:])
                   for i in range(0, len(stem) - 1, 2)]
    out_modules.append(modules[-1])
    return torch.nn.ModuleList(out_modules)


class HDDParameter(torch.nn.Parameter):
    file_name: str
    __slots__ = ['file_name']

    @staticmethod
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.zeros(())
        meta = data.new_empty((0,))
        meta.set_(meta.storage(), 0, data.size(), data.stride())
        r = torch.Tensor._make_wrapper_subclass(cls, data.size(), strides=data.stride(), device=data.device,
                                                storage_offset=data.storage_offset(), dtype=data.dtype,
                                                layout=data.layout, requires_grad=requires_grad)
        file_name = f'.temporary_tensor_buffer_{secrets.token_urlsafe(32)}.pth'
        torch.save(data, file_name)
        r.file_name = file_name
        return r

    def __repr__(self):
        return f"OffloadedParameter({self.data})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return func(*tree_map(_unwrap_offloaded_parameter, args), **tree_map(_unwrap_offloaded_parameter, kwargs))

    def __del__(self):
        os.remove(self.file_name)


def _unwrap_offloaded_parameter(inp: typing.Any) -> typing.Any:
    if not isinstance(inp, HDDParameter):
        return inp
    return torch.load(inp.file_name).requires_grad_(inp.requires_grad)


class QuantizedTensor(torch.Tensor):
    elem: torch.Tensor
    absmax: torch.Tensor
    code: torch.Tensor

    __slots__ = ['elem', 'absmax', 'code']

    @staticmethod
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.zeros(())
        meta = data.new_empty((0,))
        meta.set_(meta.storage(), 0, data.size(), data.stride())
        r = torch.Tensor._make_wrapper_subclass(cls, data.size(), strides=data.stride(), device=data.device,
                                                storage_offset=data.storage_offset(), dtype=data.dtype,
                                                layout=data.layout, requires_grad=requires_grad)
        data, (absmax, code) = quantize_blockwise(data)
        r.elem = data
        r.absmax = absmax
        r.code = code
        return r

    def __repr__(self):
        return f"QuantizedTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        out = func(*tree_map(_unwrap_offloaded_parameter, args), **tree_map(_unwrap_offloaded_parameter, kwargs))
        return tree_map(_wrap_quantized_tensor, out)


def _unwrap_quantized_tensor(inp: typing.Any) -> typing.Any:
    if not isinstance(inp, QuantizedTensor):
        return inp
    return dequantize_blockwise(inp.elem, absmax=inp.absmax, code=inp.code)


def _wrap_quantized_tensor(inp: typing.Any) -> typing.Any:
    if not isinstance(inp, torch.Tensor):
        return inp
    return QuantizedTensor(inp)
