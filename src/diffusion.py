import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from .cdm.utils.mdn_model import timestep_embedding
import math
import matplotlib.pyplot as plt

class DDIMSampler:
    def __init__(self, model, num_ddim_steps=50, eta=0.0):
        self.model = model
        self.num_ddim_steps = num_ddim_steps
        self.eta = eta  # Controls stochasticity (0 = deterministic DDIM)

    def sample(self, x_T, conditioning=None):
        """Generates samples using DDIM."""

        b, c, h, w = x_T.shape
        device = x_T.device
        img = x_T

        # Compute timesteps
        timesteps = np.linspace(0, self.model.num_timesteps - 1, self.num_ddim_steps, dtype=int)
        alphas = self.model.alphas_cumprod[timesteps]  # NumPy array
        alphas_prev = np.concatenate(([1.0], alphas[:-1]))  # NumPy array

        # Convert alphas to PyTorch tensors safely and clamp them
        alphas = torch.tensor(alphas, dtype=torch.float32, device=device).clone().detach()
        alphas_prev = torch.tensor(alphas_prev, dtype=torch.float32, device=device).clone().detach()
        alphas = torch.clamp(alphas, min=1e-5, max=1.0)
        alphas_prev = torch.clamp(alphas_prev, min=1e-5, max=1.0)

        # alphas = torch.flip(alphas, [0])
        # alphas_prev = torch.flip(alphas_prev, [0])

        # Weight for combining the two predictions (adjust as needed)
        w = 0.95

        with torch.no_grad():
            for i, step in enumerate(reversed(timesteps)):
                # Get current alphas
                alpha = alphas[i]
                alpha_prev = alphas_prev[i]

                # Compute sigma safely:
                # (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
                fraction = (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
                # Ensure non-negative values inside the sqrt:
                fraction = torch.clamp(fraction, min=0.0)
                sigma = self.eta * torch.sqrt(fraction)

                # Create timestep embedding
                step_tensor = torch.full((b,), step, device=device, dtype=torch.float32)
                t = timestep_embedding(step_tensor, self.model.time_embed_dim)

                # Pass the full two-channel image through the network
                pred_target = self.model.apply_model(img, t)
                pred_image = pred_target[:, 0:1, ...]  # Direct image prediction
                pred_noise = pred_target[:, 1:2, ...]  # Noise prediction

                # Extract the current target (noisy estimate) from the input
                current_target = img[:, 1:2, ...]

                # Compute denominators safely for the DDIM update:
                sqrt_alpha = torch.sqrt(torch.clamp(alpha, min=1e-5))
                sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1 - alpha, min=1e-5))

                # 1) Compute candidate x0 from noise prediction (standard DDIM update)
                pred_x0_from_noise = (current_target - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha

                # 2) Combine the candidate with the direct image prediction
                pred_x0 = w * pred_x0_from_noise + (1 - w) * pred_image

                # 3) Compute directional term
                sqrt_one_minus_alpha_prev = torch.sqrt(torch.clamp(1 - alpha_prev, min=1e-5))
                dir_xt = sqrt_one_minus_alpha_prev * pred_noise

                # 4) Update the target channel using the DDIM formula
                sqrt_alpha_prev = torch.sqrt(torch.clamp(alpha_prev, min=1e-5))
                noise_term = sigma * torch.randn_like(current_target)
                updated_target = sqrt_alpha_prev * pred_x0 + dir_xt + noise_term

                # 5) Reassemble the two channels:
                #    - Channel 0 (conditioning/original) remains unchanged.
                #    - Channel 1 is updated.
                img = torch.cat([img[:, 0:1, ...], updated_target], dim=1)
                img = img.detach()  # Prevent gradient accumulation

                # Check for numerical issues
                if torch.isnan(img).any() or torch.isinf(img).any():
                    print(f"🚨 NaN/Inf detected at step {step}! Breaking out of the loop.")
                    break

        return img
############################################
# TimestepBlock & TimestepEmbedSequential
############################################
class TimestepBlock(nn.Module):
    """
    A module that takes (x, t_emb) as input, for FiLM-like conditioning.
    """
    def forward(self, x, t_emb):
        raise NotImplementedError

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A Sequential that also passes t_emb to its children if they are TimestepBlock.
    """
    def forward(self, x, t_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t_emb)
            else:
                x = layer(x)
        return x

############################################
# Minimal ResBlock with time embedding (FiLM)
############################################
class ResBlock(TimestepBlock):
    def __init__(
        self,
        in_channels,
        dropout,
        out_channels=None,
        dims=2,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        down=False,
        up=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels

        # Normal conv blocks
        self.in_layers = nn.Sequential(
            normalization(in_channels),
            nn.SiLU(),
            conv_nd(dims, in_channels, self.out_channels, 3, padding=1),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip_connection = (
            nn.Conv2d(in_channels, self.out_channels, 1)
            if self.out_channels != in_channels else nn.Identity()
        )

        # For FiLM scale/shift
        # We'll project time_emb => 2*out_channels => (gamma, beta)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(192, 2 * self.out_channels),
        )

        self.down = down
        self.up = up
        if down:
            self.h_downsample = Downsample(in_channels, True, dims, out_channels=self.out_channels)
            self.x_downsample = Downsample(in_channels, True, dims, out_channels=self.out_channels)
        else:
            self.h_downsample = self.x_downsample = nn.Identity()
        if up:
            self.h_upsample = Upsample(in_channels, True, dims, out_channels=self.out_channels)
            self.x_upsample = Upsample(in_channels, True, dims, out_channels=self.out_channels)
        else:
            self.h_upsample = self.x_upsample = nn.Identity()

    def forward(self, x, t_emb):
        # If downsample, do it first
        if self.down:
            x = self.x_downsample(x)

        if self.up:
            x = self.x_upsample(x)

        # normal resblock
        h = self.in_layers(x)

        # FiLM
        # Suppose we first project t_emb to [B, out_channels*2],
        # then apply scale/shift after the first out_layers norm?
        # We'll keep it minimal.
        # We'll map t_emb => [B, self.out_channels*2]
        # But we must ensure t_emb has dimension = self.out_channels.
        # If you do a big time MLP outside, that MLP can produce exactly self.out_channels.
        # For example, in the main net's constructor, you do t_mlp => [B, model_channels], etc.
        gamma_beta = self.time_emb_proj(t_emb)  # [B, 2*out_channels]
        b, two_c = gamma_beta.shape
        c = two_c // 2
        gamma = gamma_beta[:, :c].unsqueeze(-1).unsqueeze(-1)
        beta = gamma_beta[:, c:].unsqueeze(-1).unsqueeze(-1)

        # out_layers
        # out_layers[0] is groupnorm, out_layers[1] is SiLU, out_layers[2] is Dropout, out_layers[3] is conv
        # We'll do the scale shift after the groupnorm:
        h = self.out_layers[0](h)  # groupnorm
        h = h * (1 + gamma) + beta
        h = self.out_layers[1](h)  # SiLU
        h = self.out_layers[2](h)  # Dropout
        h = self.out_layers[3](h)  # final conv

        return self.skip_connection(x) + h

############################################
# Minimal AttentionBlock ignoring time_emb
############################################
class AttentionBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels*3, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x, t_emb):
        # ignoring t_emb for attention
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H*W)
        qkv = self.qkv(h)  # [B, 3C, HW]
        q, k, v = torch.split(qkv, C, dim=1)
        q = q.reshape(B*self.num_heads, C//self.num_heads, H*W)
        k = k.reshape(B*self.num_heads, C//self.num_heads, H*W)
        v = v.reshape(B*self.num_heads, C//self.num_heads, H*W)
        attn = (q.transpose(-1, -2) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v.transpose(-1, -2)).transpose(-1, -2)
        out = out.reshape(B, C, H*W)
        out = self.proj_out(out)
        out = out.reshape(B, C, H, W)
        return x + out

############################################
# Downsample & Upsample from your code
############################################
class Downsample(nn.Module):
    def __init__(self, channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = conv_nd(dims, channels, self.out_channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

############################################
# conv_nd, zero_module, normalization (same as your code)
############################################
def conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    elif dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    elif dims == 3:
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    else:
        raise ValueError(f'unsupported dims: {dims}')

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def normalization(channels):
    return nn.GroupNorm(32, channels)

def get_alphas_cumprod(num_timesteps=1000, schedule='cosine'):
    """
    Compute the cumulative product of alphas for a DDIM (or DDPM) schedule.

    :param num_timesteps: Number of steps in the diffusion process.
    :param schedule: 'linear' or 'cosine'.
    :return: A 1-D torch.Tensor of length `num_timesteps` with alphas_cumprod.
    """
    if schedule == 'linear':
        # A typical linear schedule starts with beta1 = 1e-4 and goes up to 0.02
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
    elif schedule == 'cosine':
        # Following the improved DDPM (cosine) schedule:
        # alphas_bar = f(t), where t in [0,1], then betas = 1 - alphas_bar[i+1]/alphas_bar[i]
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
        # small offset to prevent alpha_bar from being 0
        s = 0.008
        # cosine formula from improved DDPM
        alphas_bar = torch.cos(
            ((x / num_timesteps) + s) / (1 + s) * (math.pi / 2)
        ) ** 2
        alphas_bar = alphas_bar / alphas_bar[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        betas = torch.clamp(betas, min=0, max=0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod.float()

class DDIMUNet(nn.Module):
    """
    Same structure as your UNetModel, but now each block is TimestepEmbedSequential,
    and each ResBlock can accept time embeddings. We'll define a small time embedding MLP too.
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
        time_embed_dim=192,  # dimension for learned time MLP
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.num_timesteps = 1000
        self.alphas_cumprod = get_alphas_cumprod(self.num_timesteps, schedule='cosine')

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
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_new_attention_order = use_new_attention_order

        # 1) Time embedding: first sinusoidal, then MLP => [B, time_embed_dim]
        self.time_embed_dim = time_embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, model_channels*4),
            nn.SiLU(),
            nn.Linear(model_channels*4, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            # (0) first conv => TimestepEmbedSequential ignoring t_emb
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1

        # 2) Encoder
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        dropout,
                        out_channels=int(mult*model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=False
                    )
                ]
                ch = int(mult*model_channels)
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

            if level != len(channel_mult)-1:
                out_ch = ch
                if resblock_updown:
                    down_seq = TimestepEmbedSequential(
                        ResBlock(
                            ch, dropout, out_channels=out_ch, dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True
                        )
                    )
                else:
                    down_seq = TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                ds *= 2
                self.input_blocks.append(down_seq)
                ch = out_ch
                input_block_chans.append(ch)
                self._feature_size += ch

        # 3) Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order
            ),
            ResBlock(
                ch, dropout, dims=dims, use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm
            ),
        )
        self._feature_size += ch

        # 4) Decoder
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch+ich,
                        dropout,
                        out_channels=int(model_channels*mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        up=False
                    )
                ]
                ch = int(model_channels*mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order
                        )
                    )
                if level and i==num_res_blocks:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(
                            ResBlock(
                                ch, dropout, out_channels=out_ch, dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                up=True
                            )
                        )
                    else:
                        layers.append(
                            Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                        )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # final
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps):
        """
        :param x: [N, C, ...] input
        :param timesteps: [N] 1-D array of int or float timesteps
        :return: model output
        """
        # 1) Get time embedding (sinusoidal => MLP)
        #  Suppose we want time_embed_dim=256
        #  => we do sinusoidal(timesteps, 256) => [N,256], then self.time_mlp => [N,256].
        t_emb = self.time_mlp(timesteps)  # [N, time_embed_dim], used by ResBlocks

        # 2) Encoder
        hs = []
        h = x
        for block in self.input_blocks:
            h = block(h, t_emb)  # TimestepEmbedSequential => each child gets (h, t_emb)
            hs.append(h)

        # 3) Middle
        h = self.middle_block(h, t_emb)

        # 4) Decoder
        for block in self.output_blocks:
            # skip
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.out(h)

    def apply_model(self, x, t_emb, conditioning=None):
        return self.forward(x, t_emb)

import torch.utils.data as data
import cv2

def load_image_as_tensor(img_path, bit, transform):
    """ Load an image and convert it into a tensor. """
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    # Determine bit depth (8-bit or 16-bit)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    elif image.dtype == np.uint16 and bit == 16:  # 16-bit grayscale (but contains 12-bit data)
        min_val, max_val = image.min(), image.max()
        img_tensor = torch.tensor(((image - min_val)/(max_val - min_val)), dtype=torch.float32)
    elif image.dtype == np.uint8 and bit == 8:  # Assume 8-bit grayscale
        img_tensor = torch.tensor((image/255.0), dtype=torch.float32)
    else:
        raise ValueError(f"Failed to load image, bit unmatched: {img_path}, loading {self.bit}-bit, real data type {image.dtype}")
    
    img_tensor = img_tensor.unsqueeze(0)
    
    if transform:
        return transform(img_tensor)
    
    return img_tensor

class MendeleySAGMidDataset(data.Dataset):
    data_root = "/data/spine/mendeley-lumbar/train_clean"
    valid_axes = {"SAG"}
    valid_modals = {"T1", "T2"}
    
    def __init__(self, axis: str, original_modal: str, target_modal: str, bit=8, transform=None):
        if axis not in self.valid_axes:
            raise ValueError(f"Invalid axis: {axis}. Must be one of {self.valid_axes}.")
        if original_modal not in self.valid_modals:
            raise ValueError(f"Invalid original_modal: {original_modal}. Must be one of {self.valid_modals}.")
        if target_modal not in self.valid_modals:
            raise ValueError(f"Invalid target_modal: {target_modal}. Must be one of {self.valid_modals}.")

        with open(os.path.join(self.data_root, "exclude_list.txt"), "r", encoding="utf-8") as file:
            data = file.read().strip()
            exclude_list = eval(data)

        self.axis = axis
        self.original_modal = original_modal
        self.target_modal = target_modal
        self.bit = bit
        
        mri_dirs = [
            os.path.join(patient_dir, mir_dir)
            for patient_dir in [
                os.path.join(self.data_root, patient)
                for patient in os.listdir(self.data_root)
                if patient not in exclude_list
            ]
            if os.path.isdir(patient_dir)
            for mir_dir in os.listdir(patient_dir)
        ]
        
        self.original_mri_dirs = [os.path.join(mri_dirs, 'images', f'{bit}bit') for mri_dirs in mri_dirs if original_modal in mri_dirs and axis in mri_dirs]
        self.target_mri_dirs = [os.path.join(mri_dirs, 'images', f'{bit}bit') for mri_dirs in mri_dirs if target_modal in mri_dirs and axis in mri_dirs]
        
        self.transform = transform
        
        self.len = len(self.original_mri_dirs)
        
    def __getitem__(self, idx):
        """
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]
            - Slices of PNG images as tensors (supporting 8-bit and 16-bit grayscale)
        """
        
        files = sorted(os.listdir(self.original_mri_dirs[idx]))
        if not files:
            return None
        full_path = os.path.join(self.original_mri_dirs[idx], files[(len(files)-1)//2])
        original_image = load_image_as_tensor(full_path, self.bit, self.transform)
        
        files = sorted(os.listdir(self.target_mri_dirs[idx]))
        if not files:
            return None
        full_path = os.path.join(self.target_mri_dirs[idx], files[(len(files)-1)//2])
        target_image = load_image_as_tensor(full_path, self.bit, self.transform)

        # print(original_image[0].max(), original_image[0].min())
        # print(target_image[0].max(), target_image[0].min())

        # print(original_image.shape)
        # print(target_image.shape)

        # exit()
        
        return original_image, target_image
    
    
    def __len__(self):
        return self.len

if __name__ == "__main__":
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from einops import repeat
    from .utils import MRIDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--data_dir", type=str, default="/data/datasets/spine/gtu/train")
    parser.add_argument("--original_modal", type=str, default="t1")
    parser.add_argument("--target_modal", type=str, default="t2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="/data/model")
    args = parser.parse_args()

    model_type = "diffusion"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    # Initialize Model and DDIM Sampler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDIMUNet(image_size=256, in_channels=2,
                  model_channels=96, out_channels=2, 
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 2, 4, 8]).to(device)

    # Move model parameters to the same device
    for param in model.parameters():
        param.requires_grad = True  # Ensure gradients are enabled
        param.data = param.data.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    sampler = DDIMSampler(model, num_ddim_steps=50, eta=0.0)

    # Dummy dataset of random grayscale (1,1,256,256) images
    from tqdm import tqdm
    train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    # Training Loop with DDIM Sampling
    epochs = args.num_epochs
    for epoch in range(epochs):
        for i, (original_images, target_images) in enumerate(tqdm(train_loader)):
            original_images = original_images.to(device, non_blocking=True)
            target_images = target_images.to(device, non_blocking=True)
            
            batch_size = original_images.shape[0]
            
            # Ensure all tensors are moved to GPU
            timesteps = torch.randint(0, 1000, (batch_size,), device=device).float()
            noise = torch.randn_like(target_images, device=device) # Add noise to target images

            time_embed = timestep_embedding(timesteps, 192)

            # Forward Diffusion: Add noise to `target_images`
            alpha_t = model.alphas_cumprod[timesteps.long().cpu()]  # Get alpha_t for each timestep
            sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            noisy_targets = sqrt_alpha_t * target_images + sqrt_one_minus_alpha_t * noise
            # alpha_t = 0.9
            # noisy_targets = alpha_t * target_images + (1 - alpha_t) * noise

            # Concatenate `original_images` and `noisy_target` along the channel dimension
            input_images = torch.cat([original_images, noisy_targets], dim=1)

            # Predict the denoised image from the noisy target, conditioned on `original_images`
            pred_target = model(input_images, time_embed)

            pred_noise = pred_target[:, 1].unsqueeze(1)
            pred_image = pred_target[:, 0].unsqueeze(1)

            # Compute Loss (Mean Squared Error between predicted and ground-truth `target_images`)
            loss = F.mse_loss(pred_noise, noise)
            loss_img = F.mse_loss(pred_image, target_images)

            loss = (loss + 0.5) + (loss_img * 1.5)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # # Save the model after each epoch
        # if epoch % 10 == 0 and epoch > 0:
        #     #get the predicted target images with the sampler 
        #     time_steps = torch.randint(0, 1000, (1,), device=device).float()
        #     #create a random noise for the target images
        #     noise = torch.randn_like(target_images, device=device)
        #     #concatenate the original images and the noisy target images
        #     input_images = torch.cat([original_images, noise], dim=1)

        #     sampled_images = sampler.sample(input_images, time_steps)

        #     sampled_images = sampled_images[0, 1]

        #     #save the image in a single plot
        #     plt.figure(figsize=(8, 3))  # Reduce figure size to make it more compact

        #     # Plot Original Image
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(original_images[0].cpu().numpy().squeeze(), cmap='gray')
        #     plt.axis("off")
        #     plt.title("Original", fontsize=10)

        #     # Plot Target Image
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(target_images[0].cpu().numpy().squeeze(), cmap='gray')
        #     plt.axis("off")
        #     plt.title("Target", fontsize=10)

        #     # Plot Denoised Image
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(sampled_images.cpu().numpy(), cmap='gray')
        #     plt.axis("off")
        #     plt.title("Denoised", fontsize=10)

        #     # Adjust layout for compact spacing
        #     plt.subplots_adjust(wspace=0.05, hspace=0)  # Reduce spacing between images
        #     plt.tight_layout(pad=0)  # Remove extra padding

        #     # Save the image with minimal margin
        #     plt.savefig(f"diffusion_output_{epoch}.png", bbox_inches='tight', dpi=300)
        #     plt.close()

        #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    subject_name = args.data_dir.split('/')[-3]
    dataset_name = args.data_dir.split('/')[-2]

    if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
        os.makedirs(os.path.join(args.save_dir, model_type, subject_name, dataset_name), exist_ok=True)

    # save model
    torch.save(model.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'model_{args.original_modal}_{args.target_modal}.pth'))
