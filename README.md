# RevLib

Simple and efficient RevNet-Library for PyTorch with XLA and DeepSpeed support and parameter offload

## Table of Contents

* [RevLib](#revlib)
    * [Table Of Contents](#table-of-contents)
    * [Features](#features)
    * [Getting Started](#getting-started)
        * [Installation](#installation)
    * [Examples](#examples)
        * [Reversible Backward](#reversible-backward)
        * [Parameter Offload](#parameter-offload)
        * [Coupling](#coupling)
        * [Models](#models)
            * [iRevNet](#irevnet)
            * [Reformer](#reformer)
    * [Explanation](#explanation)

## Features

* Half the constant memory usage and faster than RevNet libraries
* Less memory than gradient checkpointing (`1 * output_size` instead of `n_layers * output_size`)
* Same speed as activation checkpointing
* Extensible
* Trivial code (<100 Lines)

## Getting started

### Installation

```
python3 -m pip install revlib
```

### Examples

#### Reversible Backward

Invertible functions allow for huge memory savings as the input can be recovered from which the gradient computation can
be restarted. It's a bit like gradient checkpointing, but with recoverable inputs. That's why a reversible network
should use less memory than a network with gradient checkpointing, and both should use less maximum memory than a normal
network.

```PYTHON
import torch
from torch.utils.checkpoint import checkpoint as checkpoint_fn
import copy
import revlib

depth = 1024
batch_size = 4096

# Create network of multiple layers (so that checkpointing makes a difference) with weight sharing (cheaper)
base = [torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.ReLU(),
                            torch.nn.Linear(1, 1, bias=False))] * depth
baseline = torch.nn.Sequential(*base)
revnet = revlib.ReversibleSequential(*base)
checkpoint = base[0]


# Forcibly enable gradients so that checkpointing stores tensors
@torch.enable_grad()
def memory_utilization(mod: torch.nn.Module, checkpoint: bool = False, features: int = 1) -> int:
    torch.cuda.empty_cache()  # Empty cache, just in case PyTorch didn't do it (which is usually the case)
    mod = copy.deepcopy(mod).cuda()  # Copy model to avoid modifying the global copy. Deallocated after the function ran
    inp = torch.randn(batch_size, features, requires_grad=True).cuda()
    if not checkpoint:
        _unused = mod(inp)  # Compute a normal forward pass if not using gradient checkpointing
    else:
        for _ in range(depth):
            inp = checkpoint_fn(mod, inp)  # Manually iterate over all layers as torch doesn't support module wrapping
    return torch.cuda.memory_allocated()  # Take accurate GPU memory measurements (CPU is very inaccurate)


assert memory_utilization(baseline) > memory_utilization(baseline, True) > memory_utilization(revnet, features=2)
# Outputs: 50349056, 16794624, 99328
# 48 MiB, 16 MiB, 97 KiB
```

#### Parameter Offload

Another way to save even more memory, especially for deep networks, is to offload parameters and optimizer parameters
off the GPU onto the CPU. That way the permanent storage of the network is offloaded and only the frequently-accessed
temporary cache, used to compute the immediate gradients, is kept on the GPU.

```PYTHON
import torch
import copy
import revlib

depth = 256
width = 1024
batch_size = 1

base = [torch.nn.Linear(width, width, bias=False) for _ in range(depth)]


# Initialize network with separate weights for each layer, so that offloading has a measurable benefit


def memory_utilization(offload: bool) -> int:
    torch.cuda.empty_cache()  # Empty cache, just in case PyTorch didn't do it (which is usually the case)
    mod = copy.deepcopy(revlib.ReversibleSequential(*base, target_device="cuda" * offload))  # Copy to dealloc model
    if not offload:  # If not offloading to CPU, manually move the parameters
        mod = mod.cuda()
    _unused = mod(torch.randn(batch_size, width * 2).cuda())  # Normal forward pass, 2x features because of RevNet
    return torch.cuda.memory_allocated()  # Take accurate GPU memory measurements (CPU is very inaccurate)


assert memory_utilization(False) > memory_utilization(True)
# Outputs: 1073750016, 8192
# 1 GiB, 8 KiB
```

Another way of doing parameter offload would be to manually call `revlib.offload_tensor` before accessing each parameter
of a custom model. This way, you can control when the parameters are loaded onto the GPU. Sometimes it's faster to load
everything onto the GPU in a single operation, such as before calling a TorchScript function, while other times it's
more memory-efficient to load every parameter seconds before its usage.\
Internally, RevLib has the offload_tensor functionality integrated into its reversible core, giving a faster experience
thanks to parallel `non_blocking` operations.

#### Coupling

Another major feature of RevLib is to use custom coupling functions such as the one used in
[MomentumNet](https://arxiv.org/abs/2102.07870). It's a recent paper that made significant advancements in the area of
memory-efficient networks. They propose to use a momentum stream instead of a second model output as illustrated
below: ![MomentumNetIllustration](http://limitless.sh/momentumnet.png)
<p align="center">Image from <a href=https://twitter.com/PierreAblin/status/1426899071495819265>the plagiarized</a> <a href=https://arxiv.org/abs/2108.05862v2>mRevNet</a></p>

Using a custom coupling operation (the functional analogue of [MemCNN](https://github.com/silvandeleemput/memcnn)) that
merges input and output streams, MomentumNet can be implemented in RevLib as seen below:

```PYTHON
import torch
from torch import nn
import revlib

channels = 64
depth = 16
momentum_ema_beta = 0.99


# Compute y2 from x2 and f(x1) by merging x2 and f(x1) in the forward pass.
def momentum_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return other_stream * momentum_ema_beta + fn_out * (1 - momentum_ema_beta)


# Calculate x2 from y2 and f(x1) by manually computing the inverse of momentum_coupling_forward.
def momentum_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return (output - fn_out * (1 - momentum_ema_beta)) / momentum_ema_beta


# Pass in coupling functions which will be used instead of x2 + f(x1) and y2 - f(x1)
rev_model = revlib.ReversibleSequential(*[layer for _ in range(depth)
                                          for layer in [nn.Conv2d(channels, channels, (3, 3), padding=1),
                                                        nn.Identity()]],
                                        coupling_forward=[momentum_coupling_forward, revlib.additive_coupling_forward],
                                        coupling_inverse=[momentum_coupling_inverse, revlib.additive_coupling_inverse])

inp = torch.randn((16, channels * 2, 224, 224))
out = rev_model(inp)
assert out.size() == (16, channels * 2, 224, 224)
```

When implementing MomentumNet like this, there is no storage for lost information in the forward pass which the
MomentumNet paper accounts for. One way to work around that issue is to avoid the coupling function
altogether. [HomebrewNLP integrated the coupling functions into f() and g()](https://github.com/HomebrewNLP/HomebrewNLP/blob/efda4b1dbc320c620ed024208f0745b82fb30ebf/src/model.py#L209-L232)
which means that there is no loss of information, no matter the depth or beta of the model.

#### Models

##### iRevNet

[iRevNet](https://openreview.net/forum?id=HJsjkMb0Z) is not only partially reversible but instead a fully-invertible
model. The [source code](https://github.com/jhjacobsen/pytorch-i-revnet) looks complex at first glance. It also doesn't
use the memory savings it could utilize, as RevNet requires custom AutoGrad functions that are hard to maintain. An
iRevNet can be implemented like this using revlib:

```PYTHON
import torch
from torch import nn
import revlib

channels = 64
channel_multiplier = 4
depth = 3
classes = 1000


# Create a basic function that's reversibly executed multiple times. (Like f() in ResNet)
def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)


def block_conv(in_channels, out_channels):
    return nn.Sequential(conv(in_channels, out_channels),
                         nn.Dropout(0.2),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


def block():
    return nn.Sequential(block_conv(channels, channels * channel_multiplier),
                         block_conv(channels * channel_multiplier, channels),
                         nn.Conv2d(channels, channels, (3, 3), padding=1))


# Create a reversible model. f() is invoked depth-times with different weights.
rev_model = revlib.ReversibleSequential(*[block() for _ in range(depth)])

# Wrap reversible model with non-reversible layers
model = nn.Sequential(conv(3, 2 * channels), rev_model, conv(2 * channels, classes))

# Use it like you would a regular PyTorch model
inp = torch.randn((1, 3, 224, 224))
out = model(inp)
out.mean().backward()
assert out.size() == (1, 1000, 224, 224)
```

##### Reformer

[Reformer](https://arxiv.org/abs/2001.04451) uses RevNet with chunking and LSH-attention to efficiently train a
transformer. Using revlib, standard implementations, such
as [lucidrains' Reformer](https://github.com/lucidrains/reformer-pytorch/), can be improved upon to use less memory.
Below we're still using the basic building blocks from lucidrains' code to have a comparable model.

```PYTHON
import torch
from torch import nn
from reformer_pytorch.reformer_pytorch import LSHSelfAttention, Chunk, FeedForward, AbsolutePositionalEmbedding
import revlib


class Reformer(torch.nn.Module):
    def __init__(self, sequence_length: int, features: int, depth: int, heads: int, bucket_size: int = 64,
                 lsh_hash_count: int = 8, ff_chunks: int = 16, input_classes: int = 256, output_classes: int = 256):
        super(Reformer, self).__init__()
        self.token_embd = nn.Embedding(input_classes, features * 2)
        self.pos_embd = AbsolutePositionalEmbedding(features * 2, sequence_length)

        self.core = revlib.ReversibleSequential(*[nn.Sequential(nn.LayerNorm(features), layer) for _ in range(depth)
                                                  for layer in
                                                  [LSHSelfAttention(features, heads, bucket_size, lsh_hash_count),
                                                   Chunk(ff_chunks, FeedForward(features, activation=nn.GELU),
                                                         along_dim=-2)]],
                                                split_dim=-1)
        self.out_norm = nn.LayerNorm(features * 2)
        self.out_linear = nn.Linear(features * 2, output_classes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.out_linear(self.out_norm(self.core(self.token_embd(inp) + self.pos_embd(inp))))


sequence = 1024
classes = 16
model = Reformer(sequence, 256, 6, 8, output_classes=classes)
out = model(torch.ones((16, sequence), dtype=torch.long))
assert out.size() == (16, sequence, classes)
```

## Explanation

Most other RevNet libraries, such as [MemCNN](https://github.com/silvandeleemput/memcnn)
and [Revtorch](https://github.com/RobinBruegger/RevTorch) calculate both f() and g() in one go, to create one large
computation. RevLib, on the other hand, brings Mesh
TensorFlow's ["reversible half residual and swap"](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/layers.py#L2191)
to PyTorch. `reversible_half_residual_and_swap` computes only one of f() and g() and swaps the inputs and gradients.
This way, the library only has to store one output as it can recover the other output during the backward pass.\
Following Mesh TensorFlow's example, revlib also uses separate x1 and x2 tensors instead of concatenating and splitting
at every step to reduce the cost of memory-bound operations.

RevNet's memory consumption doesn't scale with its depth, so it's significantly more memory-efficient for deep models.
One problem in most implementations was that two tensors needed to be stored in the output, quadrupling the required
memory. The high memory consumption rendered RevNet nearly useless for small networks, such as BERT, with its six
layers.\
RevLib works around this problem by storing only one output and two inputs for each forward pass, giving a model as
small as BERT a >2x improvement!

Ignoring the dual-path structure of a RevNet, it usually used to be much slower than gradient checkpointing. However,
RevLib uses minimal coupling functions and has no overhead between Sequence items, allowing it to train as fast as a
comparable model with gradient checkpointing.

