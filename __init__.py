import typing

import torch

QUAD_TENSOR = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class _ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp0: torch.Tensor, inp1: torch.Tensor, tmp_inp0: torch.Tensor, tmp_inp1: torch.Tensor):
        ctx.save_for_backward(tmp_inp0)
        return inp0, inp1

    @staticmethod
    def backward(ctx, grad0: torch.Tensor, grad1: torch.Tensor):
        tmp_inp0, = ctx.saved_tensors
        return grad0, tmp_inp0, grad1, torch.zeros_like(tmp_inp0)


class _ReversibleHalfResidualSwapFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x0: torch.Tensor, back_x0: torch.Tensor, x1: torch.Tensor, back_x1: torch.Tensor,
                mod: torch.nn.Module) -> QUAD_TENSOR:
        ctx.mod = mod
        return x1, back_x0, x0 + mod(x1), back_x1

    @staticmethod
    def backward(ctx, dy0: torch.Tensor, y0: torch.Tensor, dy1: torch.Tensor, y1: torch.Tensor
                 ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        with torch.enable_grad():
            y0 = y0.requires_grad_(True)
            out = ctx.mod(y0)
        with torch.no_grad():
            x0 = y1 - out.detach()
        with torch.enable_grad():
            dx0, *param_grad = torch.autograd.grad(out, (y0,) + tuple(ctx.mod.parameters()), dy1)
        with torch.no_grad():
            for p, g in zip(ctx.mod.parameters(), param_grad):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad.data.add_(g)
        with torch.enable_grad():
            return dy1.detach(), x0.detach(), dx0.add(dy0).detach(), y0.detach(), None


replace_grad = _ReplaceGrad().apply
reverse_and_swap = _ReversibleHalfResidualSwapFn().apply


class ReversibleModule(torch.nn.Module):
    def __init__(self, wrapped_module: torch.nn.Module):
        super(ReversibleModule, self).__init__()
        self.wrapped_module = wrapped_module

    def forward(self, inp: QUAD_TENSOR) -> QUAD_TENSOR:
        return reverse_and_swap(*inp, self.wrapped_module)


class ReversibleSequential(torch.nn.Module):
    def __init__(self, *modules, split_dim=1):
        super(ReversibleSequential, self).__init__()
        self.stem = torch.nn.Sequential(*[m if isinstance(m, ReversibleModule) else ReversibleModule(m)
                                          for m in modules])
        self.split_dim = split_dim

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp0, inp1 = inp.chunk(2, self.split_dim)
        zeros = torch.zeros_like(inp0)
        return torch.cat(replace_grad(*self.stem((inp0, zeros, inp1, zeros))), dim=self.split_dim)
