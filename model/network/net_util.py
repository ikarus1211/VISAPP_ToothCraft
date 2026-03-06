"""
Author: David Pukanec (FIT BUT)

This code was produced by David Pukanec (FIT VUT) and was inspired by:
  - DiffComplete: https://github.com/dvlab-research/DiffComplete
  - ControlNet:   https://github.com/lllyasviel/ControlNet
"""


from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from xformers.ops import memory_efficient_attention


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.
    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, cond=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, EfficientCrossAttentionBlock):
                x = layer(x, cond, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv3d(channels, channels, 3, padding=1)


    def forward(self, x):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv3d(channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool3d(3, stride=stride,)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        activation = nn.SiLU()
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            activation,
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            activation,
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            activation,
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 3, padding=1 )
        else:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h



class EfficientCrossAttentionBlock(nn.Module):
    def __init__(self, q_channels: int, kv_channels: int, final_channels: int,tensor_res, emb_channels, heads: int = 8, ):
        super().__init__()
        assert final_channels % heads == 0, "Channels must be divisible by number of heads"
        self.channels = final_channels
        self.heads = heads
        self.head_dim = final_channels // heads

        self.rope = RoPENd([tensor_res,tensor_res,tensor_res, final_channels])
        self.q_proj = nn.Conv1d(q_channels, final_channels, kernel_size=1)
        self.kv_proj = nn.Conv1d(kv_channels, final_channels*2, kernel_size=1)
        self.proj_out = zero_module(nn.Conv1d(final_channels, final_channels, 1))
        self.norm = nn.LayerNorm(final_channels)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                final_channels
            ),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, t) -> torch.Tensor:
        """
        x: (B, C, D, H, W) — input to be transformed
        context: (B, S, context_dim) — conditioning sequence (e.g., text, image tokens)
        """
        B, C, D, H, W = x.shape
        N = D * H * W

        # Reshape: (B, C, D, H, W)
        x = x.view(B, C, N)
        cB, cC, cD, cH, cW = context.shape
        cond = context.view(cB, cC, cD * cH * cW)
        q =  self.q_proj(x)
        q = q + self.emb_layers(t).unsqueeze(-1)
        kv = self.kv_proj(cond)
        # (B, C * 2, N)
        k, v = kv.chunk(2, dim=1)

        # ( B, C, N) -> (B, N, C)
        q = self.rope(q.permute(0, 2, 1))
        k = self.rope(k.permute(0, 2, 1))
        v = v.permute(0, 2, 1)

        # Split heads: (B, N, heads, head_dim)
        def split_heads(t):
            return t.view(B, N, self.heads, self.head_dim)

        q, k, v = map(split_heads, (q, k, v))

        # Memory-efficient attention
        out = memory_efficient_attention(q.contiguous(), k.contiguous(), v.contiguous())  # (B * heads, N, head_dim)

        # Merge heads: (B , N, head_dim *  self.heads) -> (B, N, C)
        out = out.view(B, N, self.heads * self.head_dim)
        out = out.permute(0, 2, 1)
        # Output projection
        out = self.proj_out(out)  # (B, C, N)

        # Reshape back to (B, C, D, H, W)
        out = out.view(B, C, D, H, W)
        return out


class EfficientSelfAttentionBlock(nn.Module):

    def __init__(self, channels: int, feat_res, heads: int = 8):
        super().__init__()
        assert channels % heads == 0, "Channels must be divisible by number of heads"
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads

        self.rope = RoPENd([feat_res,feat_res,feat_res,channels])
        self.qkv_proj = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W

        # Reshape: (B, C, D, H, W)
        x = x.view(B, C, N)
        #x = self.norm(x)

        # Linear projection to q, k, v
        qkv = self.qkv_proj(x)  # (B, C * 3, N)
        q, k, v = qkv.chunk(3, dim=1)

        q = self.rope(q.permute(0, 2, 1))
        k = self.rope(k.permute(0, 2, 1))
        v = v.permute(0, 2, 1)

        # Split heads: (B, N, C) -> (B, N, heads, head_dim)
        def split_heads(t):
            return t.view(B, N, self.heads, self.head_dim)

        q, k, v = map(split_heads, (q, k, v))

        # Memory-efficient attention
        out = memory_efficient_attention(q.contiguous(), k.contiguous(), v.contiguous())  # (B * heads, N, head_dim)

        # Merge heads: (B , N, head_dim *  self.heads) -> (B, N, C)
        out = out.view(B, N, self.heads * self.head_dim)
        out = out.permute(0, 2, 1)
        # Output projection
        out = self.proj_out(out)  # (B, C, N)

        # Reshape back to (B, C, D, H, W)
        out = out.view(B, C, D, H, W)
        return out

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.
        Meant to be used like:
            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class RoPENd(nn.Module):
    """N-dimensional Rotary Positional Embedding."""
    def __init__(self, shape, base=10000):
        super(RoPENd, self).__init__()

        tensor_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(tensor_dims))
        self.num_dims = len(tensor_dims)
        self.k_max = k_max
        # tensor of angles to use
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))

        # create a stack of angles multiplied by position
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in
                            torch.meshgrid([torch.arange(d) for d in tensor_dims], indexing='ij')], dim=-1)

        # convert to complex number to allow easy rotation
        rotations = torch.polar(torch.ones_like(angles), angles)

        # store in a buffer so it can be saved in model parameters
        self.register_buffer('rotations', rotations.view(-1, rotations.shape[-1]))

    def forward(self, x):
        # convert input into complex numbers to perform rotation
        expected_dim = 2 * self.num_dims * self.k_max
        x_rot_part = x[..., :expected_dim]
        x_rest = x[..., expected_dim:] if x.shape[-1] > expected_dim else None

        x_rot_complex = torch.view_as_complex(x_rot_part.reshape(*x.shape[:-1], -1, 2).contiguous())
        pe_x = self.rotations * x_rot_complex
        pe_x_real = torch.view_as_real(pe_x).flatten(-2).to(x.dtype)

        if x_rest is not None:

            return torch.cat([pe_x_real, x_rest], dim=-1)
        else:
            return pe_x_real
