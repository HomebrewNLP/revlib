import typing

import torch


def additive_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return other_stream + fn_out


def additive_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output - fn_out


class ReversibleSequential(torch.nn.Module):
    def __init__(self, *modules, split_dim=1,
                 coupling_forward: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None,
                 coupling_inverse: typing.Optional[typing.List[typing.Optional[typing.Callable]]] = None):
        super(ReversibleSequential, self).__init__()
        self.split_dim = split_dim
        self.module_list = torch.nn.ModuleList(modules)
        self.coupling_forward = list(coupling_forward) if coupling_forward else [additive_coupling_forward]
        self.coupling_inverse = list(coupling_inverse) if coupling_inverse else [additive_coupling_inverse]

        self.cpu_state: torch.Tensor = torch.get_rng_state()
        self.cuda: bool = torch.cuda._initialized
        self.autocast: bool = torch.is_autocast_enabled()

        self.x0: typing.Optional[torch.Tensor] = None
        self.x1: typing.Optional[torch.Tensor] = None
        self.idx: int = 0
        self.mod_idx: int = 0
        self.counter: int = 0
        self.storage: typing.Dict[str, torch.Tensor] = {}

    def get_key(self, idx: int, inp: torch.Tensor):
        key = f'Index: {idx}\nSize: {inp.size()}\nDevice: {inp.device}\nDataType: {inp.dtype}'
        return key

    def pack(self, inp: torch.Tensor) -> str:
        self.counter += 1
        return self.get_key(self.counter - 1, inp)

    def inner_pack(self, inp: torch.Tensor):
        self.storage[self.get_key(len(self.storage), inp)] = inp

    def inner_unpack(self, key: str):
        raise RuntimeError(f'Tensor not found.\nSpec:\n{key}')

    def unpack(self, key: str) -> torch.Tensor:
        if self.storage:
            if key not in self.storage:
                self.inner_unpack(key)
            return self.storage[key]

        rng_devices = []
        if self.cuda:
            rng_devices = self.devices
        with torch.random.fork_rng(devices=rng_devices):
            torch.set_rng_state(self.cpu_state)
            with torch.enable_grad(), torch.cuda.amp.autocast(self.autocast):
                with torch.autograd.graph.saved_tensors_hooks(self.inner_pack, self.inner_unpack):
                    out = self.module_list[self.idx](self.x1)
                x1 = self.x1
                x0 = self.x0
                self.x1 = self.coupling_inverse[self.mod_idx](self.x0, out.detach())
                self.x0 = x1
                with torch.autograd.graph.saved_tensors_hooks(self.inner_pack, self.inner_unpack):
                    _unused = self.coupling_forward[self.mod_idx](x0, out)
        return self.unpack(key)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        self.x0, self.x1 = inp.chunk(2, self.split_dim)
        for self.idx in range(len(self.module_list)):
            self.mod_idx = self.idx % len(self.coupling_forward)
            self.counter = 0
            self.storage = {}
            x0, x1 = self.x0, self.x1
            with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
                y1 = self.coupling_forward[self.mod_idx](x0, self.module_list[self.idx](x1))
            self.x1 = y1
            self.x0 = x1

        return torch.cat([self.x0, self.x1], dim=self.split_dim)
