import os
import math
import torch
import torch.nn as nn
import numpy as np
from .mdn_model import timestep_embedding

class CDMDDIMSampler:
    def __init__(self, model, mdn_model, num_ddim_steps=50, eta=0.0):
        self.model = model
        self.mdn_model = mdn_model
        self.num_ddim_steps = num_ddim_steps
        self.eta = eta  # Controls stochasticity (0 = deterministic DDIM)

    def sample(self, x_T, conditioning=None):
        """Generates samples using DDIM."""
        b, c, h, w = x_T.shape
        device = x_T.device
        img = x_T

        # Compute timesteps
        timesteps = np.linspace(0, self.mdn_model.num_timesteps - 1, self.num_ddim_steps, dtype=int)
        alphas = self.mdn_model.alphas_cumprod[timesteps]
        alphas_prev = np.concatenate(([1.0], alphas[:-1]))

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

                pred_noise = self.mdn_model.apply_model(conditioning, step)

                # Compute x_0 (predicted denoised image)
                pred_x0 = (conditioning - torch.sqrt(torch.clamp(1 - alpha, min=1e-5)) * pred_noise) / torch.sqrt(torch.clamp(alpha, min=1e-5))

                # Compute x_{t-1}
                dir_xt = torch.sqrt(1 - alpha_prev) * pred_noise
                noise = sigma * torch.randn_like(conditioning)
                conditioning = torch.sqrt(alpha_prev) * pred_x0 + dir_xt + noise
                conditioning = conditioning.detach()
        
        with torch.no_grad():
            img = self.model(img, conditioning)

        return img
