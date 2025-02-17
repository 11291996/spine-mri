import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float())


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


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


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


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


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :]  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, k=None, v=None):

        attention_tensors = {}

        for i, layer in enumerate(self):
            if isinstance(layer, TimestepBlock):
                x = layer(x)
            elif isinstance(layer, AttentionBlock):
                x, k, v = layer(x, k, v)
                attention_tensors[f"k_{i}"] = k
                attention_tensors[f"v_{i}"] = v
            else:
                x = layer(x)
        return x, attention_tensors

    def cross_forward(self, x, attention_tensors):
        for i, layer in enumerate(self):
            if isinstance(layer, TimestepBlock):
                x = layer(x)
            elif isinstance(layer, AttentionBlock):
                k = attention_tensors[f"k_{i}"]
                v = attention_tensors[f"v_{i}"]
                x, k, v = layer(x, k, v)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
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

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

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
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, ), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        # emb_out = self.emb_layers(emb)
        # while len(emb_out.shape) < len(h.shape):
        #     emb_out = emb_out[..., None]

        # rdm_rep_out = self.emb_layers_rdm(rdm_rep)
        # while len(rdm_rep_out.shape) < len(h.shape):
        #     rdm_rep_out = rdm_rep_out[..., None]
        
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=4,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, k, v):
        return checkpoint(self._forward, (x, k, v,), self.parameters(), True)

    def _forward(self, x, k, v):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h, k, v = self.attention(qkv, k, v)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial), k, v


def count_flops_attn(model, _x, y):
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
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, key=None, value=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        if key is not None and value is not None:
            k, v = key, value
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length), k, v

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1))
        )

    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        h = x

        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
    
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h)

        return self.out(h)


    def image_encode(self, x):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []

        h = x

        for module in self.input_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        features = h 
        features = h.reshape(h.shape[0], h.shape[1], -1)
        features = features.mean(dim=2)

        return features

class CondMLP(nn.Module):
    def __init__(self, input_dim=192, hidden_dim=256, output_dim=192):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, cond):
        # cond: [batch, 192]
        return self.fc(cond).unsqueeze(-1).unsqueeze(-1)

class CrossCUNet(nn.Module):
    """
    A variant of UNetModel that uses a separate CondMLP for each block:
    - Each downsample block in `input_blocks`
    - The middle block
    - Each block in `output_blocks`
    Then we add the CondMLP output (per-block) to the feature map
    before passing it into that block.
    """
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # Extra CondMLP config:
        cond_in_dim=192,
        cond_hidden_dim=256,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order


        # --- Build the input blocks (encoder) ---
        ch = int(channel_mult[0] * model_channels)
        ds = 1
        self.input_blocks = nn.ModuleList()
        self.cond_mlps_input = nn.ModuleList()  # same length as input_blocks

        # Block 0: first conv
        block0 = TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        self.input_blocks.append(block0)
        # We'll create a CondMLP for block0 that outputs 'ch'
        self.cond_mlps_input.append(CondMLP(cond_in_dim, cond_hidden_dim, in_channels))

        input_block_chans = [ch]

        # Loop over channel_mult levels
        for level, mult in enumerate(channel_mult):
            # repeated ResBlocks
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                seq_rb = TimestepEmbedSequential(*layers)
                self.input_blocks.append(seq_rb)
                # each res/attn block gets its own CondMLP => out = ch
                self.cond_mlps_input.append(CondMLP(cond_in_dim, cond_hidden_dim, 1))
                input_block_chans.append(ch)
            # Downsample if not last level
            if level != len(channel_mult) - 1:
                out_ch = ch
                if resblock_updown:
                    down_seq = TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                    )
                else:
                    down_seq = TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                ds *= 2
                self.input_blocks.append(down_seq)
                # cond mlp for this downsample => out_ch = ch
                self.cond_mlps_input.append(CondMLP(cond_in_dim, cond_hidden_dim, ch))
                ch = out_ch
                input_block_chans.append(ch)

        # --- Middle block ---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        # Single cond mlp for entire middle block => out dim=ch
        self.cond_mlp_middle = CondMLP(cond_in_dim, cond_hidden_dim, ch)

        # --- Output blocks (decoder) ---
        self.output_blocks = nn.ModuleList()
        self.cond_mlps_output = nn.ModuleList()
        # We'll replicate the same logic
        self._feature_size = 0
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(
                            ResBlock(
                                ch,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True,
                            )
                        )
                    else:
                        layers.append(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                    ds //= 2
                seq_ = TimestepEmbedSequential(*layers)
                self.output_blocks.append(seq_)
                # add cond mlp for each output block => out=ch
                self.cond_mlps_output.append(CondMLP(cond_in_dim, cond_hidden_dim, ch))

        # final out
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, int(channel_mult[0]*model_channels), out_channels, 3, padding=1)),
            nn.Tanh()
        )

    def forward(self, x, cond):
        """
        x: [B, in_channels, H, W]
        cond: [B, cond_in_dim] => each block has separate cond mlp => [B, block_channels,1,1]
        We do: x = x + cond_emb BEFORE passing x to that block.
        """
        # 1) input blocks
        hs = []
        h = x.to(self.dtype)

        attention_tensors = {}

        for i, block in enumerate(self.input_blocks):
            # each block has a matching cond mlp
            cond_emb = self.cond_mlps_input[i](cond)  # [B, block_out_ch,1,1]
            # pass through block
            h, attention_tensors_in = block(h)
            attention_tensors[f"input_block_{i}"] = attention_tensors_in
            h = h + cond_emb
            hs.append(h)

        # 2) middle block
        cond_emb_mid = self.cond_mlp_middle(cond)  # [B, ch,1,1]
        h = h + cond_emb_mid
        h, attention_tensors_mid = self.middle_block(h)
        attention_tensors[f"middle_block"] = attention_tensors_mid
    
        # 3) output blocks
        for i, block in enumerate(self.output_blocks):
            if hs:
                skip = hs.pop()
                h = th.cat([h, skip], dim=1)
            cond_emb_out = self.cond_mlps_output[i](cond)
            h, attention_tensors_out = block(h)
            attention_tensors[f"output_block_{i}"] = attention_tensors_out
            if i == len(self.output_blocks) - 1:
                pass
            else:
                h = h + cond_emb_out

        return self.out(h), attention_tensors

    def cross_attention_forward(self, x, cond, attention_tensors):
        # 1) input blocks
        hs = []
        h = x.to(self.dtype)

        for i, block in enumerate(self.input_blocks):
            # each block has a matching cond mlp
            cond_emb = self.cond_mlps_input[i](cond)
            # pass through block
            h, _ = block(h)
            h = h + cond_emb
            hs.append(h)

        # 2) middle block
        cond_emb_mid = self.cond_mlp_middle(cond)
        h = h + cond_emb_mid
        h = self.middle_block.cross_forward(h, attention_tensors["middle_block"])

        # 3) output blocks
        for i, block in enumerate(self.output_blocks):
            if hs:
                skip = hs.pop()
                h = th.cat([h, skip], dim=1)
            cond_emb_out = self.cond_mlps_output[i](cond)
            h = block.cross_forward(h, attention_tensors[f"output_block_{i}"])
            if i == len(self.output_blocks) - 1:
                pass
            else:
                h = h + cond_emb_out

        return self.out(h)

from src.cdm.utils.mdn_model import SimpleMLP, timestep_embedding

class CrossCDMDDIMSampler:
    def __init__(self, input_model, target_model, input_mdn_model, target_mdn_model, num_ddim_steps=50, eta=0.0):
        self.input_model = input_model
        self.target_model = target_model
        self.input_mdn_model = input_mdn_model
        self.target_mdn_model = target_mdn_model
        self.num_ddim_steps = num_ddim_steps
        self.eta = eta  # Controls stochasticity (0 = deterministic DDIM)

    def sample(self, x_T_input, x_T_target, conditioning=None):
        """Generates samples using DDIM."""
        b, c, h, w = x_T.shape
        device = x_T.device
        img_input = x_T_input
        img_target = x_T_target

        # Compute timesteps
        timesteps = np.linspace(0, self.input_mdn_model.num_timesteps - 1, self.num_ddim_steps, dtype=int)
        alphas = self.input_mdn_model.alphas_cumprod[timesteps]
        alphas_prev = np.concatenate(([1.0], alphas[:-1]))

        # inversion conditioning

        inversion_noises = []

        #copy conditioning

        conditioning_input = conditioning.clone()

        #get the noise from the input_mdn_model
        with torch.no_grad():
            for i, step in enumerate(timesteps):
                alpha = alphas[i]
                if not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
                alpha_prev = alphas_prev[i]
                if not isinstance(alpha_prev, torch.Tensor):
                    alpha_prev = torch.tensor(alpha_prev, device=device, dtype=torch.float32)
                sigma = self.eta * torch.sqrt(
                    torch.clamp((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev), min=1e-5)
                )
                #create tensor of the step
                step = torch.full((b, ), step, device=device, dtype=torch.float32)

                pred_noise = self.input_mdn_model.apply_model(conditioning_input, step)

                # Compute x_0 (predicted denoised image)
                pred_x0 = (conditioning_input - torch.sqrt(torch.clamp(1 - alpha, min=1e-5)) * pred_noise) / torch.sqrt(torch.clamp(alpha, min=1e-5))

                # Compute x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
                noise = sigma * torch.randn_like(conditioning_input)
                conditioning_input = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
                conditioning_input = conditioning_input.detach()
                inversion_noises.append(noise)

        with torch.no_grad():
            for i, step in enumerate(reversed(timesteps)):
                alpha = alphas[i]
                if not isinstance(alpha, torch.Tensor):
                    alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
                alpha_prev = alphas_prev[i]
                if not isinstance(alpha_prev, torch.Tensor):
                    alpha_prev = torch.tensor(alpha_prev, device=device, dtype=torch.float32)
                sigma = self.eta * torch.sqrt(
                    torch.clamp((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev), min=1e-5)
                )
                #create tensor of the step
                step = torch.full((b, ), step, device=device, dtype=torch.float32)

                pred_noise = self.target_mdn_model.apply_model(conditioning, step)

                # add inversion noise
                gamma = 0.5
                pred_noise = pred_noise * (1 - gamma) + inversion_noises[i] * gamma

                # Compute x_0 (predicted denoised cond)
                pred_x0 = (conditioning - torch.sqrt(torch.clamp(1 - alpha, min=1e-5)) * pred_noise) / torch.sqrt(torch.clamp(alpha, min=1e-5))

                # Compute x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
                noise = sigma * torch.randn_like(conditioning)
                conditioning = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
                conditioning = conditioning.detach()
        
        with torch.no_grad():
            _, attention_tensors = self.model_input(img_input, conditioning_input)
            img = self.model_target.cross_attention_forward(img_target, conditioning, attention_tensors)

        return img

if __name__ == "__main__":
    import os
    import torch
    import argparse
    from ..utils import MRIDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--input_model_dir", type=str, default="/data/model/cdm/brain/brat2024/model_t1_t2.pth")
    parser.add_argument("--target_model_dir", type=str, default="/data/model/cdm/spine/cdm/model_t1_t2.pth")
    parser.add_argument("--datasets_dir", type=str, default="/data/datasets/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from torch.utils.data import DataLoader
    from src.gan_score import torch_psnr, torch_ssim_masked, torch_mae, torch_lpips

    original_modal = args.model_dir.split("/")[-1].split("_")[1]
    target_modal = args.model_dir.split("/")[-1].split("_")[2].split(".")[0]

    input_subject_name = args.input_model_dir.split("/")[-3]
    input_dataset_name = args.input_model_dir.split("/")[-2]

    input_data_dir = os.path.join(args.datasets_dir, input_subject_name, input_dataset_name, "test")

    # Test Dataset 
    input_test_dataset =  MRIDataset(input_data_dir, original_modal, target_modal)
    input_test_loader = DataLoader(dataset=test_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=False)
    input_model = CrossCUNet(image_size=256, in_channels=1,
                      model_channels=96, out_channels=1, 
                      num_res_blocks=1, attention_resolutions=[32,16,8],
                      channel_mult=[1, 1, 2, 2]).cuda()
    input_mdn_model = SimpleMLP(in_channels=192, time_embed_dim= 192, model_channels=1536, bottleneck_channels=1536, out_channels=192, num_res_blocks=12).cuda()
    input_model.load_state_dict(torch.load(args.input_model_dir))
    input_mdn_model.load_state_dict(torch.load(args.input_model_dir.replace("model", "mdn")))

    target_subject_name = args.target_model_dir.split("/")[-3]
    target_dataset_name = args.target_model_dir.split("/")[-2]

    target_data_dir = os.path.join(args.datasets_dir, target_subject_name, target_dataset_name, "test")
    
    target_test_dataset =  MRIDataset(target_data_dir, original_modal, target_modal)
    target_test_loader = DataLoader(dataset=target_test_dataset, num_workers=0, batch_size=1, shuffle=False)
    target_model = CrossCUNet(image_size=256, in_channels=1,
                        model_channels=96, out_channels=1, 
                        num_res_blocks=1, attention_resolutions=[32,16,8],
                        channel_mult=[1, 1, 2, 2]).cuda()
    target_mdn_model = SimpleMLP(in_channels=192, time_embed_dim= 192, model_channels=1536, bottleneck_channels=1536, out_channels=192, num_res_blocks=12).cuda()
    target_model.load_state_dict(torch.load(args.target_model_dir))
    target_mdn_model.load_state_dict(torch.load(args.target_model_dir.replace("model", "mdn")))

    sampler = CrossCDMDDIMSampler(input_model, target_model, input_mdn_model, target_mdn_model)

    psnr, ssim, mae, lpips, count = 0.0, 0.0, 0.0, 0.0, 0

    loss_fn = lpips.LPIPS(net="alex").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    for i, (input_image, target_image) in enumerate(zip(input_test_loader, target_test_loader)):
        input_model_input = input_image[0] 
        target_model_input = target_image[0]
        target_image = target_image[1]

        random_cond = torch.randn(1, 192, device=device)

        output_image = sampler.sample(input_model_input, target_model_input, random_cond)

        output_image = output_image[:, 0]
        target_image = target_image.squeeze(0)

        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
        target_image = (target_image - target_image.min()) / (target_image.max() - target_image.min())

        psnr += torch_psnr(output_image, target_image)
        ssim += torch_ssim_masked(output_image, target_image)
        mae += torch_mae(output_image, target_image)
        lpips += torch_lpips(output_image.cuda(), target_image.cuda(), loss_fn)
        count += 1

    print(f"PSNR: {psnr/count}, SSIM: {ssim/count}, MAE: {mae/count}, LPIPS: {lpips/count}")
