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
        * [Utils](#utils)
            * [HuggingFace](#huggingface-integration)
            * [Cast Intermediates](#Cast-Intermediates)
            * [Offload Intermediates](#Offload-Intermediates)
    * [Explanation](#explanation)
    * [Citation](#citation)

## Features

* Less memory than gradient checkpointing (`2 * output_size` instead of `n_layers * output_size`)
* Same speed as activation checkpointing
* Extensible
* Native HuggingFace, DeepSpeed, and XLA support

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
which means that there is no loss of information, no matter the depth or beta of the model. The same integrated
MomentumNet is available via the [utils module](#utils).

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

#### Merged Optimizer

Another optimization RevLib allows is to merge the optimizer step and backward.\
Instead of first computing a backward pass and then applying the gradients in a separate stage, RevLib can apply the
gradients immediately while calculating the backward pass. This change allows you to get speedups by taking advantage of
asynchronous computation and means that you don't have to instantiate all gradients simultaneously. So, instead of
storing all gradients simultaneously, you only keep the gradients of one layer while still arriving at the same results.
Below is a small example demonstrating just how much memory this can save:

```PYTHON
import random
import typing

import numpy as np
import torch

import revlib


def optim(params: typing.Iterable[torch.Tensor]):
    return torch.optim.SGD(params, lr=1e-3)


SIZE = 2048
DEPTH = 64
BATCH_SIZE = 1
STEPS = 4


def block():
    return torch.nn.Sequential(torch.nn.Linear(SIZE, SIZE),
                               torch.nn.ReLU(),
                               torch.nn.Linear(SIZE, SIZE))


def run(fused: bool):
    torch.manual_seed(42069)
    random.seed(42069)
    np.random.seed(42069)
    model = revlib.ReversibleSequential(*[block() for _ in range(DEPTH)], fused_optimizer=optim if fused else None)
    model.cuda()

    optimizer = None if fused else optim(model.parameters())
    mean_loss = 0
    max_mem = 0
    for i in range(STEPS):
        max_mem = max(torch.cuda.memory_allocated(), max_mem)
        inp = torch.randn((BATCH_SIZE, SIZE * 2), requires_grad=True, device='cuda')
        max_mem = max(torch.cuda.memory_allocated(), max_mem)
        loss = (model(inp) - inp).abs().mean()
        max_mem = max(torch.cuda.memory_allocated(), max_mem)
        loss.backward()
        max_mem = max(torch.cuda.memory_allocated(), max_mem)
        if not fused:
            optimizer.step()
            max_mem = max(torch.cuda.memory_allocated(), max_mem)
            model.zero_grad(set_to_none=True)
        max_mem = max(torch.cuda.memory_allocated(), max_mem)
        with torch.no_grad():
            mean_loss += loss.item()
    print(f"Loss: {mean_loss / STEPS:12.10f} - Memory: {max_mem * 2 ** -20:7.2f} MiB")


run(True)  # Fused Optimizer. Results:       Loss: 1.7444351912 - Memory: 2049.05 MiB
run(False)  # Default Optimizer. Results:    Loss: 1.7444351912 - Memory: 4098.03 MiB
```

As you can see, while the loss is still the exact same, the model uses half the memory at its peak. The freed-up memory
would allow you to create 504 million more parameters. Considering that the model only has 512 million parameters, this
would mean you could use ~100% more parameters!\
Of course, the absolute freed memory would stay the same if the optimizer had buffers, such as SGD with momentum.
Because of that, the relative memory advantage would decrease. That's why a memory-efficient optimizer
like [SM3](https://arxiv.org/abs/1901.11150) or
[8-bit Adam](https://github.com/facebookresearch/bitsandbytes/#using-the-8-bit-optimizers) is perfect here.

#### Utils

##### HuggingFace Integration

RevLib also has its own `utils` module which provides helpful functions as `residual_to_momentum_net`. Using RevLib, you
can trivially convert any HuggingFace transformer into a MomentumNet without significant loss of performance. Especially
during fine-tuning, this can be a life-saver, as it allows for significantly bigger models to fit into memory without
the need to manually (or [automatically](https://arxiv.org/abs/2006.09616)) create countless buffers for activation
checkpointing.\
With the addition of `MomentumNet`, there is one more hyperparameter to tune. Small values of `beta` allow the model to
continue functioning as normal:

```PYTHON
from transformers import AutoModelForCausalLM, GPT2TokenizerFast
from revlib.utils import module_list_to_momentum_net

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokens = tokenizer(["The shadowy hacker group Eleuther"], return_tensors='pt')['input_ids']
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
original_layers = model.transformer.h
print(tokenizer.decode(model.generate(input_ids=tokens)[0]))
# The shadowy hacker group Eleutheria has been busy for the past few months. The group has been

model.transformer.h = module_list_to_momentum_net(original_layers, residual=True, beta=0.1)
print(tokenizer.decode(model.generate(input_ids=tokens)[0]))
# The shadowy hacker group Eleutheria has been busy for the past few months. The group has been
```

On the other hand, large values improve numerical stability of deep networks at the cost of slightly altering the
information flow.

```PYTHON
model.transformer.h = module_list_to_momentum_net(original_layers, residual=True, beta=0.5)
print(tokenizer.decode(model.generate(input_ids=tokens)[0]))
# The shadowy hacker group Eleutherian psi- psi- psi- psi psi psi psi psi psi psi
```

Either way, both can be used to train the models just the same as you're used to! While the gradients might differ
between models, there is no performance degradation after fine-tuning.

```PYTHON
model(tokens)[0].mean().backward()
print(next(iter(model.parameters())).grad.mean().item())
# -7.596428730494154e-08
```

As expected, the memory consumption for the modified model is significantly lower during training than that of a
non-checkpointed model:

```PYTHON
import time
import torch
from transformers import AutoModelForCausalLM
from memory_profiler import memory_usage

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
tokens = torch.zeros((4, 2048), dtype=torch.long)
start_time = time.time()
memory = max(memory_usage((lambda: model(tokens)[0].mean().backward(),)))
print(time.time() - start_time)
# 206.94576001167297
print(memory - max(memory_usage((lambda: None,))))
# 150272.09765625
```

```PYTHON
import time
import torch
from transformers import AutoModelForCausalLM
from revlib.utils import module_list_to_momentum_net
from memory_profiler import memory_usage

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
model.transformer.h = module_list_to_momentum_net(model.transformer.h, residual=True, beta=0.5)  # The only difference
tokens = torch.zeros((4, 2048), dtype=torch.long)
start_time = time.time()
memory = max(memory_usage((lambda: model(tokens)[0].mean().backward(),)))
print(time.time() - start_time)
# 275.42114186286926
print(memory - max(memory_usage((lambda: None,))))
# 6187.0703125
```

##### Cast Intermediates

Another nice feature RevLib has, is that it can automatically offload intermediate values or cast them to another
datatype. Casting the intermediate tensors used during the backward pass from float32 to half-precision (float16) would
halve the memory required for intermediate values.\
To integrate it into your existing code, you only have to wrap your step with the `memory_efficient_intermediates`
context manager. Wrapping multiple steps or even the entire training loop would cause memory leaks. Below you can see a
small example of how the optimisation could use it in practice:

```PYTHON
import torch

from revlib.utils import memory_efficient_intermediates

ITEMS = 2 ** 24
ITERATIONS = 32


def run():
    # Model code here
    torch.manual_seed(0)
    out = a = torch.randn((ITEMS,), device='cuda', dtype=torch.float32, requires_grad=True)
    for i in range(ITERATIONS):
        out = out * torch.randn_like(out)
    print(f'Output: {out.mean().item():} - Memory: {torch.cuda.memory_allocated() * 1e-6:.2f}MB', end='')
    out.mean().backward()
    print(f' - Grad: {a.grad.mean().item()}')


run()  # Output: -0.0002206185890827328 - Memory: 2281.70MB - Grad: 0.00011316053132759407

with memory_efficient_intermediates(torch.half):
    run()  # Output: -0.0002206185890827328 - Memory: 1207.96MB - Grad: 0.00011316053132759407
```

The peak memory consumption is over 2GB when running the function normally, as PyTorch has to allocate many intermediate
values and store them in float32. If you instead add a cast to the tensors kept for the backward pass, the memory
consumption gets halved while both output and gradient stay the same. Here, we only have to add
the `memory_efficient_intermediates` context wrapper, which handles casts automatically. Note that this only changes the
tensors that are kept for the backward pass and alters the gradients slightly but doesn't influence the forward pass in
any way. Nevertheless, doing it this way is critical to avoid casting down to float16 and back up again during the
forward pass.\
Similar to casts from float32 to float16, you could also cast float64 to float16, float64 to float32 or even mix these!\
For example, when switching the computation datatype above from float32 to float64, the program will generate the
following printout:

```
Output: 0.00019801305610731704 - Memory: 4563.40MB - Grad: -1.1288092155692513e-11
Output: 0.00019801305610731704 - Memory: 1342.18MB - Grad: -1.1293845258815898e-11
```

As you can see, the model uses almost four times less memory, giving you the memory advantages of float16 without losing
any of the precision of float64.

##### Offload Intermediates

Going one step further with the concepts from above, we can even offload the intermediate values onto the CPU.
Intermediate-Offloading is akin [Parameter-Offloading](#parameter-offload), as we've done above, but moves the adaptive
memory onto the CPU while the GPU is free to compute whatever it wants. In practice, moving all intermediates means that
the model has the same GPU-memory consumption as if it were to run with `torch.no_grad` or in `torch.inference_mode`
while still allowing backpropagation without any loss of accuracy!

```PYTHON
import torch

from revlib.utils import memory_efficient_intermediates

ITEMS = 2 ** 24
ITERATIONS = 32


def run():
    # Model code here
    torch.manual_seed(0)
    out = a = torch.randn((ITEMS,), device='cuda', dtype=torch.float32, requires_grad=True)
    for i in range(ITERATIONS):
        out = out * torch.randn_like(a)
    print(f'Output: {a.mean().item():} - Memory: {torch.cuda.memory_allocated() * 1e-6:.2f}MB', end='')
    out.mean().backward()
    print(f' - Grad: {out.mean().item()}')


run()  # Output: 0.00011316053132759407 - Memory: 2281.70MB - Grad: 1.1337489974616588e-11

with memory_efficient_intermediates(storage_device='cpu'):  # <-- This is the only line that's modified
    run()  # Output: 0.00011316053132759407 - Memory: 134.22MB - Grad: 1.1337489974616588e-11

with torch.no_grad():
    run()  # Output: 0.00011316053132759407 - Memory: 134.22MB
    # It will error here, but that doesn't matter as the memory gets measured before the backward pass.
```

As you can see, the new memory consumption is the same as if the model were to run with `torch.no_grad`, with the minor
difference that it can still produce 100% accurate gradients. Of course, this free memory doesn't come from anywhere.
It's just that the tensors that have to be stored in the normal computation (but not with `torch.no_grad`)
are now moved to the CPU.\
However, as there is no real prefetching, the model will be slower, as PyTorch has to query a buffer for every
intermediate tensor used in the backward pass and get the tensors from there.

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

## Citation

```BIBTEX
@software{nestler_lucas_2022_6837352,
  author       = {Nestler, Lucas},
  title        = {RevLib},
  month        = jul,
  year         = 2022,
  publisher    = {Zenodo},
  version      = {1.7.0},
  doi          = {10.5281/zenodo.6837352},
  url          = {https://doi.org/10.5281/zenodo.6837352}
}```