# RevLib

Simple and efficient RevNet-Library with DeepSpeed support

## Features

* Half the constant memory usage of other RevNet libraries
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

#### Sequential CNN like iRevNet

[iRevNet](https://openreview.net/forum?id=HJsjkMb0Z) is not on partially reversible, but instead a fully-invertible
model. The [original source code](https://github.com/jhjacobsen/pytorch-i-revnet) looks complex at first glance. It also
doesn't use the memory-savings it could utilize, as RevNet requires custom AutoGrad functions that are hard to maintain.
Using revlib, an iRevNet can be implemented like this:

```PYTHON
import torch
from torch import nn
import revlib

channels = 64
channel_multiplier = 4
depth = 16
classes = 1000


# Create basic function that's executed multiple times in a reversible way. (Like f() in ResNet)
def conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, (3, 3), padding=1)


def block_conv(in_channels, out_channels):
    return nn.Sequential(conv(in_channels, out_channels),
                         nn.Dropout(0.2),
                         nn.BatchNorm2d(channels),
                         nn.ReLU())


def block():
    return nn.Sequential(block_conv(channels, channels * channel_multiplier),
                         block_conv(channels * channel_multiplier, channels),
                         nn.Conv2d(channels, channels, (3, 3), padding=1))


# Create reversible model. f() is invoked depth-times with different weights.
rev_model = revlib.ReversibleSequential(*[block() for _ in range(depth)])

# Wrap reversible model with non-reversible layers
model = nn.Sequential(nn.Conv2d(3, 2 * channels, (3, 3)),
                      rev_model,
                      nn.Conv2d(2 * channels, classes, (3, 3)))

# Use it like you would a normal PyTorch model
inp = torch.randn((16, 3, 224, 224))
out = model(inp)
assert out.size() == (16, 1000, 224, 224)
```

#### MomentumNet

[MomentumNet](https://arxiv.org/abs/2102.07870) is another recent paper that made great advancements in the area of
memory-efficient networks. They propose to use a momentum stream instead of a second model output as illustrated
below: ![MomentumNetIllustration](http://limitless.sh/momentumnet.png). Implementing that with revlib requires you to
write a custom coupling operation (functional analogue to [MemCNN](https://github.com/silvandeleemput/memcnn)) that
merges input and output streams.

```PYTHON
import torch
from torch import nn
import revlib

channels = 64
depth = 16
momentum_ema_beta = 0.99


# Compute x2_{t+1} from x2_{t} and f(x1_{t}) by merging x2 and f(x1) in the forward pass.
def momentum_coupling_forward(other_stream: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return other_stream * momentum_ema_beta + fn_out * (1 - momentum_ema_beta)


# Calculate x2_{t} from x2_{t+1} and f(x1_{t}) by manually computing the inverse of momentum_coupling_forward.
def momentum_coupling_inverse(output: torch.Tensor, fn_out: torch.Tensor) -> torch.Tensor:
    return output / momentum_ema_beta - fn_out * (1 - momentum_ema_beta)


# Pass in coupling functions which will be used instead of x2_{t} + f(x1_{t}) and x2_{t+1} - f(x1_{t})
rev_model = revlib.ReversibleSequential(*[nn.Conv2d(channels, channels, (3, 3), padding=1) for _ in range(depth)],
                                        coupling_forward=momentum_coupling_forward,
                                        coupling_inverse=momentum_coupling_inverse)

inp = torch.randn((16, channels * 2, 224, 224))
out = rev_model(inp)
assert out.size() == (16, channels * 2, 224, 224)
```

#### Reformer

[Reformer](https://arxiv.org/abs/2001.04451) uses RevNet, together with chunking and LSH-attention to efficiently
calculate train a transformer. Using revlib, common implementations, such
as [lucidrains' Reformer](https://github.com/lucidrains/reformer-pytorch/), can be improved upon to use less memory.
Below we're still using the basic building blocks from lucidrains' code, to have a comparable model.

```PYTHON
import torch
from torch import nn
from reformer_pytorch.reformer_pytorch import LSHSelfAttention, Chunk, FeedForward, AbsolutePositionalEmbedding
import revlib


class Reformer(torch.nn.Module):
    def __init__(self, sequence_length: int, features: int, depth: int, heads: int, bucket_size: int = 64,
                 lsh_hash_count: int = 8, ff_chunks: int = 128, input_classes: int = 256, output_classes: int = 256):
        super(Reformer, self).__init__()
        self.token_embd = nn.Embedding(input_classes, features * 2)
        self.pos_embd = AbsolutePositionalEmbedding(features, sequence_length)

        self.core = revlib.ReversibleSequential([nn.Sequential(nn.LayerNorm(features), layer) for _ in range(depth)
                                                 for layer in
                                                 [LSHSelfAttention(features, heads, bucket_size, lsh_hash_count),
                                                  Chunk(ff_chunks, FeedForward(features, activation=nn.GELU))]])
        self.out_norm = nn.LayerNorm(features * 2)
        self.out_linear = nn.Linear(features * 2, output_classes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.out_linear(self.out_norm(self.core(self.token_embd(inp) + self.pos_embd)))


sequence = 1024
classes = 16
model = Reformer(sequence, 256, 6, 8, output_classes=classes)
out = model(torch.ones((16, sequence)))
assert out.size() == (16, sequence, classes)
```
