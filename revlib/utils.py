import typing

import torch
from torch.utils._pytree import tree_map

from revlib.core import (FUSED_OPTIMIZER, MemoryModes, MergeCalls, ReversibleSequential, SingleBranchReversibleModule,
                         get_key, split_tensor_list)


class MomentumNetSide(torch.nn.Module):
    def __init__(self, alpha: float):
        """
        Side-network of a MomentumNet. This part adds the current residual stream to the "velocity" stream.
        :param alpha: Scale for the residual stream. In the simplest case, it'd be (1 - beta), but it gets more
        complicated with later layers.
        """
        super(MomentumNetSide, self).__init__()
        self.alpha = alpha

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return inp * self.alpha


class MomentumNetStem(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module, gamma: float):
        """
        Main/Stem network for a MomentumNet. It calls the wrapped_module with its respective input and ensures that the
        input scale is correct by multiplying it with gamma.
        :param wrapped_module: nn.Module
        :param gamma: constant scale to normalize the input back into the correct range
        """
        super(MomentumNetStem, self).__init__()
        self.wrapped_module = wrapped_module
        self.gamma = gamma

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.wrapped_module(inp * self.gamma, *args, **kwargs)


class ResidualToPlain(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module):
        """
        A simple module that subtracts the input value from the output of `f(x)`. This is useful when replacing
        an existig residual stream with a reversible residual stream without touching the model itself.
        :param wrapped_module: nn.Module
        """
        super(ResidualToPlain, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, inp: torch.Tensor, *args, **kwargs) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
        out = split_tensor_list(self.wrapped_module(inp, *args, **kwargs))
        if isinstance(out, torch.Tensor):
            return out - inp
        return [out[0] - inp] + out[1]


def optional_detach(inp: typing.Any) -> typing.Any:
    if isinstance(inp, torch.Tensor):
        return inp.detach()
    return inp


def detached_additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor
                                       ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
    fn_out = split_tensor_list(fn_out)
    if isinstance(fn_out, torch.Tensor):
        return other_stream + fn_out
    return [other_stream + fn_out[0]] + tree_map(optional_detach, fn_out[1])


def detached_additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor
                                       ) -> typing.Union[typing.List[torch.Tensor], torch.Tensor]:
    fn_out = split_tensor_list(fn_out)
    if isinstance(fn_out, torch.Tensor):
        return output - fn_out
    return [output - fn_out[0]] + tree_map(optional_detach, fn_out[1])


def momentum_net(*modules, split_dim=1,
                 coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 memory_mode: MemoryModes = MemoryModes.autograd_function,
                 target_device: str = "",
                 fused_optimizer: FUSED_OPTIMIZER = None,
                 beta: float = 0.9) -> ReversibleSequential:
    """
    Creates a sequential MomentumNet by wrapping each layer in MomentumNet-Wrappers and dispatching to
    ReversibleSequential

    :param modules: All nn.Modules that should be wrapped. It's the same syntax as nn.Sequential, but adds a
    reversible residual stream.
    :param split_dim: RevNets require two streams. This parameter specifies which dimension to split in half to
    create the two streams. `None` would mean the input gets replicated for both streams. It's usually best to split
    along the features, which is why the default (1) is compatible with convolutions.
    :param coupling_forward: RevNet uses y0 = (x0 + f(x1)) as a coupling function, but this allows you to set a
    custom one. For example, MomentumNet (https://arxiv.org/abs/2102.07870) uses
    y0 = (beta * x0 + (1 - beta) * f(x1)). The inputs to the coupling function are the residual stream and the
    function output. For more information, look at the examples. default = revnet couplint
    :param coupling_inverse: The inverse of the coupling function. default = revnet inverse
    :param memory_mode: One of `MemoryModes`'s values. Some things are only supported in one mode while others
    might only be supported in another. default = autograd function (highest coverage but spotty XLA support)
    :param target_device: Specifies where the parameters should be moved to before computing the forward and
    backward pass. This allows efficient CPU-offloading.
    default = no offloading (keep parameters on the device they're on)
    :param fused_optimizer: Allows an optimizer step to run while the model is computing its backward pass. This
    means that the gradients don't have to be fully instantiated anymore and can improve speed when used with
    cpu-offload due to asynchronous compute. It expects a function that generates an optimizer from a list of
    parameters. (like Adam.__init__) default = no fused optimizer step
    :param beta: MomentumNet beta value that controls how much of the velocity stream is kept.
    :return: Instantiated MomentumNet (instance of `ReversibleSequential`)
    """
    momentum_modules = []
    for idx, mod in enumerate(modules):
        momentum_modules.append(MomentumNetStem(mod, beta ** idx))
        momentum_modules.append(MomentumNetSide((1 - beta) / beta ** (idx + 1)))
    return ReversibleSequential(*momentum_modules, split_dim=split_dim, coupling_forward=coupling_forward,
                                coupling_inverse=coupling_inverse, memory_mode=memory_mode, target_device=target_device,
                                fused_optimizer=fused_optimizer)


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
                         fused_optimizer: FUSED_OPTIMIZER = None,
                         residual: bool = False) -> ReversibleSequential:
    """
    Creates a sequential RevNet by unrolling a nn.Sequential module and dispatching to `ReversibleSequential`

    :param module: An existing nn.Sequential module that should be converted into a ReversibleSequential module.
    :param split_dim: RevNets require two streams. This parameter specifies which dimension to split in half to
    create the two streams. `None` would mean the input gets replicated for both streams. It's usually best to split
    along the features, which is why the default (1) is compatible with convolutions.
    :param coupling_forward: RevNet uses y0 = (x0 + f(x1)) as a coupling function, but this allows you to set a
    custom one. For example, MomentumNet (https://arxiv.org/abs/2102.07870) uses
    y0 = (beta * x0 + (1 - beta) * f(x1)). The inputs to the coupling function are the residual stream and the
    function output. For more information, look at the examples. default = revnet couplint
    :param coupling_inverse: The inverse of the coupling function. default = revnet inverse
    :param memory_mode: One of `MemoryModes`'s values. Some things are only supported in one mode while others
    might only be supported in another. default = autograd function (highest coverage but spotty XLA support)
    :param target_device: Specifies where the parameters should be moved to before computing the forward and
    backward pass. This allows efficient CPU-offloading.
    default = no offloading (keep parameters on the device they're on)
    :param fused_optimizer: Allows an optimizer step to run while the model is computing its backward pass. This
    means that the gradients don't have to be fully instantiated anymore and can improve speed when used with
    cpu-offload due to asynchronous compute. It expects a function that generates an optimizer from a list of
    parameters. (like Adam.__init__) default = no fused optimizer step
    :param residual: Whether to "undo" a residual stream or not. Using y = f(x0) + x0 + x1 is generally not a good idea,
    so this would subtract `x0` from y allowing you to patch existing residual modules without modifying their code.
    :return: Instantiated RevNet (instance of `ReversibleSequential`)
    """
    return ReversibleSequential(*maybe_residual_to_plain(module, residual), split_dim=split_dim,
                                coupling_forward=coupling_forward, coupling_inverse=coupling_inverse,
                                memory_mode=memory_mode, target_device=target_device, fused_optimizer=fused_optimizer)


def sequential_to_momentum_net(module: torch.nn.Sequential,
                               split_dim=1,
                               coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                               coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                               memory_mode: MemoryModes = MemoryModes.autograd_function,
                               target_device: str = "",
                               fused_optimizer: FUSED_OPTIMIZER = None,
                               residual: bool = False,
                               beta: float = 0.9) -> ReversibleSequential:
    """
    Creates a sequential MomentumNet by unrolling a nn.Sequential module and dispatching to `momentum_net()`

    :param module: An existing nn.Sequential module that should be converted into a ReversibleSequential module.
    :param split_dim: RevNets require two streams. This parameter specifies which dimension to split in half to
    create the two streams. `None` would mean the input gets replicated for both streams. It's usually best to split
    along the features, which is why the default (1) is compatible with convolutions.
    :param coupling_forward: RevNet uses y0 = (x0 + f(x1)) as a coupling function, but this allows you to set a
    custom one. For example, MomentumNet (https://arxiv.org/abs/2102.07870) uses
    y0 = (beta * x0 + (1 - beta) * f(x1)). The inputs to the coupling function are the residual stream and the
    function output. For more information, look at the examples. default = revnet couplint
    :param coupling_inverse: The inverse of the coupling function. default = revnet inverse
    :param memory_mode: One of `MemoryModes`'s values. Some things are only supported in one mode while others
    might only be supported in another. default = autograd function (highest coverage but spotty XLA support)
    :param target_device: Specifies where the parameters should be moved to before computing the forward and
    backward pass. This allows efficient CPU-offloading.
    default = no offloading (keep parameters on the device they're on)
    :param fused_optimizer: Allows an optimizer step to run while the model is computing its backward pass. This
    means that the gradients don't have to be fully instantiated anymore and can improve speed when used with
    cpu-offload due to asynchronous compute. It expects a function that generates an optimizer from a list of
    parameters. (like Adam.__init__) default = no fused optimizer step
    :param residual: Whether to "undo" a residual stream or not. Using y = f(x0) + x0 + x1 is generally not a good idea,
    so this would subtract `x0` from y allowing you to patch existing residual modules without modifying their code.
    :param beta: MomentumNet beta value that controls how much of the velocity stream is kept.
    :return: Instantiated MomentumNet (instance of `ReversibleSequential`)
    """
    return momentum_net(*maybe_residual_to_plain(module, residual), split_dim=split_dim,
                        coupling_forward=coupling_forward, coupling_inverse=coupling_inverse, memory_mode=memory_mode,
                        target_device=target_device, beta=beta, fused_optimizer=fused_optimizer)


def module_list_to_momentum_net(module: torch.nn.ModuleList,
                                split_dim=1,
                                coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                                coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                                memory_mode: MemoryModes = MemoryModes.autograd_function,
                                target_device: str = "",
                                fused_optimizer: FUSED_OPTIMIZER = None,
                                residual: bool = False,
                                beta: float = 0.9) -> torch.nn.ModuleList:
    """
    Creates a sequential MomentumNet by unrolling a nn.ModuleList module. This method ensures that the inputs and
    outputs stay consistent with what they used to be, allowing it to be used as a drop-in replacement for existing
    nn.ModuleLists if they are called sequentially.

    :param module: An existing nn.Sequential module that should be converted into a ReversibleSequential module.
    :param split_dim: RevNets require two streams. This parameter specifies which dimension to split in half to
    create the two streams. `None` would mean the input gets replicated for both streams. It's usually best to split
    along the features, which is why the default (1) is compatible with convolutions.
    :param coupling_forward: RevNet uses y0 = (x0 + f(x1)) as a coupling function, but this allows you to set a
    custom one. For example, MomentumNet (https://arxiv.org/abs/2102.07870) uses
    y0 = (beta * x0 + (1 - beta) * f(x1)). The inputs to the coupling function are the residual stream and the
    function output. For more information, look at the examples. default = revnet couplint
    :param coupling_inverse: The inverse of the coupling function. default = revnet inverse
    :param memory_mode: One of `MemoryModes`'s values. Some things are only supported in one mode while others
    might only be supported in another. default = autograd function (highest coverage but spotty XLA support)
    :param target_device: Specifies where the parameters should be moved to before computing the forward and
    backward pass. This allows efficient CPU-offloading.
    default = no offloading (keep parameters on the device they're on)
    :param fused_optimizer: Allows an optimizer step to run while the model is computing its backward pass. This
    means that the gradients don't have to be fully instantiated anymore and can improve speed when used with
    cpu-offload due to asynchronous compute. It expects a function that generates an optimizer from a list of
    parameters. (like Adam.__init__) default = no fused optimizer step
    :param residual: Whether to "undo" a residual stream or not. Using y = f(x0) + x0 + x1 is generally not a good idea,
    so this would subtract `x0` from y allowing you to patch existing residual modules without modifying their code.
    :param beta: MomentumNet beta value that controls how much of the velocity stream is kept.
    :return: MomentumNet modules as `nn.ModuleList`
    """
    net = momentum_net(*maybe_residual_to_plain(module, residual), split_dim=split_dim,
                       coupling_forward=coupling_forward, coupling_inverse=coupling_inverse, memory_mode=memory_mode,
                       target_device=target_device, beta=beta)
    secondary_branch_buffer = []
    stem = list(net)[:-1]  # Drop last `MomentumNetSide`
    modules = [SingleBranchReversibleModule(secondary_branch_buffer, wrapped_module=mod.wrapped_module.wrapped_module,
                                            coupling_forward=mod.wrapped_module.coupling_forward,
                                            coupling_inverse=mod.wrapped_module.coupling_inverse,
                                            memory_savings=mod.memory_savings, target_device=mod.target_device,
                                            cache=mod.cache, first=idx == 0, last=idx == len(stem) - 1,
                                            fused_optimizer=fused_optimizer)
               for idx, mod in enumerate(stem)]
    out_modules = [MergeCalls(modules[i], modules[i + 1], collate_fn=lambda y, x: [y] + x[0][1:])
                   for i in range(0, len(stem) - 1, 2)]
    out_modules.append(modules[-1])
    return torch.nn.ModuleList(out_modules)


def is_float(inp: torch.Tensor):
    return inp.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def memory_efficient_intermediates(storage_dtype: typing.Optional[torch.dtype] = None,
                                   storage_device: typing.Optional[torch.device] = None,
                                   tensor_filter: typing.Callable[[torch.Tensor], bool] = is_float
                                   ) -> torch.autograd.graph.saved_tensors_hooks:
    """
    This function returns a context manager which turns all storage for the backward pass into a specified data type.
    By default, it will convert all float tensors to half precision, but it can also cast to any other data type like
    float32 or int8.
    In the forward pass, it will still use the tensors as they are computed, to avoid casting up and back down again.
    This ensures that the forward pass has maximum precision and does not suffer from this context manager.
    However, in the backward pass, the
    whatever data
    :param storage_dtype: Specifies the datatype used in storage. torch.half is recommended
    :param storage_device: Specifies the location where intermediate tensors are stored. Useful for CPU-Offloading of
    :param tensor_filter: A function that specifies whether a tensor should be acted upon (return True) or not
    (return False). By default, it will check if the input tensor is of float type and cast/offload it only if it is.
    intermediate values. (Not to be confused with CPU-Offloading of parameters)
    :return: A `torch.autograd.graph.saved_tensors_hooks` context manager that will improve memory efficiency
    """
    counter = 0
    storage: typing.Dict[str, torch.Tensor] = {}
    dtypes: typing.Dict[str, torch.dtype] = {}
    devices: typing.Dict[str, torch.dtype] = {}

    def pack(inp: torch.Tensor):
        nonlocal counter
        counter += 1
        assigned_name = get_key(counter - 1, inp)
        dtypes[assigned_name] = inp.dtype
        devices[assigned_name] = inp.device
        if tensor_filter(inp):
            inp = inp.to(dtype=storage_dtype, device=storage_device, non_blocking=True)
        storage[assigned_name] = inp
        return assigned_name

    def unpack(key: str) -> torch.Tensor:
        return storage[key].to(dtype=dtypes[key], device=devices[key], non_blocking=True)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)
