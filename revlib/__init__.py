import typing

import torch

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor):
        ctx.save_for_backward(tmp_inp0.detach(), tmp_inp1.detach())
        return inp0, inp1

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        tmp_inp0, tmp_inp1 = ctx.saved_tensors
        return grad0, tmp_inp0, grad1, tmp_inp1


class _ReversibleHalfResidualSwapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, back_x0: torch.Tensor, x1: torch.Tensor, back_x1: torch.Tensor,
                mod: typing.Callable, coupling_forward: typing.Callable,
                coupling_inverse: typing.Callable) -> QUAD_TENSOR:
        ctx.mod = mod
        ctx.coupling_inverse = coupling_inverse
        ctx.forward_rng_state = torch.get_rng_state()
        return x1, back_x0, coupling_forward(x0, mod(x1)), back_x1

    @staticmethod
    def backward(ctx, dy0: torch.Tensor, y0: torch.Tensor, dy1: torch.Tensor, y1: torch.Tensor
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        original_rng_state = torch.get_rng_state()
        torch.set_rng_state(ctx.forward_rng_state)
        with torch.enable_grad():
            y0 = y0.requires_grad_()
            y0.retain_grad()
            out = ctx.mod(y0)
        with torch.no_grad():
            x0 = ctx.coupling_inverse(y1, out.detach())
        torch.autograd.backward(out, dy1)
        torch.set_rng_state(original_rng_state)
        with torch.enable_grad():
            return dy1.detach(), x0.detach_(), y0.grad.add_(dy0).detach_(), y0.detach_(), None, None, None


replace_grad = _ReplaceGrad().apply
reverse_and_swap = _ReversibleHalfResidualSwapFn().apply


def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return other_stream + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - fn_out


class ReversibleModule(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module, coupling_forward: typing.Optional[typing.Callable] = None,
                 coupling_inverse: typing.Optional[typing.Callable] = None):
        super(ReversibleModule, self).__init__()
        self.wrapped_module = wrapped_module
        self.coupling_forward = coupling_forward or additive_coupling_forward
        self.coupling_inverse = coupling_inverse or additive_coupling_inverse

    def forward(self, inp: QUAD_TENSOR) -> QUAD_TENSOR:
        return reverse_and_swap(*inp, self.wrapped_module, self.coupling_forward, self.coupling_inverse)


class ReversibleSequential(torch.nn.Module):
    def __init__(self, *modules, split_dim=1,
                 coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None):
        super(ReversibleSequential, self).__init__()
        coupling_forward = list(coupling_forward) if coupling_forward else [None]
        coupling_inverse = list(coupling_inverse) if coupling_inverse else [None]
        self.stem = torch.nn.Sequential(*[m if isinstance(m, ReversibleModule) else
                                          ReversibleModule(m,
                                                           coupling_forward[i % len(coupling_forward)],
                                                           coupling_inverse[i % len(coupling_inverse)])
                                          for i, m in enumerate(modules)])
        self.split_dim = split_dim

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp0, inp1 = inp.chunk(2, self.split_dim)
        zeros = torch.zeros_like(inp0)
        return torch.cat(replace_grad(*self.stem((inp0, zeros, inp1, zeros))), dim=self.split_dim)
