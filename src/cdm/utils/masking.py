import torch
import numpy as np
from einops import rearrange

def patchify(in_channels, imgs, patch_size):
    """
    imgs: (N, 4, D, H, W)
    x: (N, L, patch_size**3 *4)
    """
    p = patch_size[0]
    assert imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_channels, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_channels))
    return x

def unpatchify(in_channels, x, patch_size, image_size):
    """
    x: (N, L, patch_size**3 *4)
    imgs: (N, 4, D, H, W)
    """
    p = patch_size[0]
    h, w = image_size
    
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_channels))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], in_channels, h * p, h * p))
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

def mask_func(x, in_channels, mask_ratio, patch_size, image_size, mask_value=0.0):
    batch = x.shape[0]
    x_patch = patchify(in_channels, x, patch_size)

    mask_patch, mask, id = random_masking(x_patch, mask_ratio)
    mask_tokens = torch.ones(1, 1, in_channels * patch_size[0] * patch_size[1]) * mask_value
    device = x.device
    mask_tokens = mask_tokens.repeat(batch,  id.shape[1] - mask_patch.shape[1], 1)
    mask_tokens = mask_tokens.to(device)

    x_ = torch.cat([mask_patch, mask_tokens], dim=1)  # no cls token
    x_ = torch.gather(x_, dim=1, index=id.unsqueeze(-1).repeat(1, 1, mask_patch.shape[2]))  # unshuffle
    # mask the input
    x = unpatchify(in_channels, x_, patch_size=patch_size, image_size=image_size)
    del mask_tokens 
    
    return x, mask