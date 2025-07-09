import torch
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return np.array(betas, dtype=np.float64)

class DDPMPipeline:
    def __init__(self, schedule_name="cosine", num_timesteps=1000, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_timesteps = num_timesteps

        if schedule_name == "linear":
            self.betas = torch.linspace(1e-4, 1e-2, num_timesteps, device=self.device)
        elif schedule_name == "cosine":
            self.betas = torch.tensor(
                betas_for_alpha_bar(
                    num_timesteps,
                    lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                ),
                dtype=torch.float32,
                device=self.device
            )

        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0).to(self.device)  # Alphas cumulative product

        # Additional precomputed values 
        self.alphas_cumprod = self.alphas_hat
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

    def forward_diffusion(self, images, timesteps) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to the input images at timestep t.
        """
        batch_size = images.shape[0]
        
        # Ensure all of these tensors are on the same device
        # Move timesteps to same device as self.alphas_cumprod
        timesteps = timesteps.to(self.alphas_cumprod.device)

        # Extract alpha_cumprod for the given timesteps
        alpha_cumprod_t = self.alphas_cumprod[timesteps]
        alpha_cumprod_t = alpha_cumprod_t.to(images.device).view(batch_size, 1, 1, 1)

        # Generate Gaussian noise
        gaussian_noise = torch.randn_like(images, device=images.device)

        # Add noise to the images using forward diffusion formula
        noised_image = torch.sqrt(alpha_cumprod_t) * images + torch.sqrt(1 - alpha_cumprod_t) * gaussian_noise

        return noised_image, gaussian_noise

    def predict_x0(self, x_t, t, noise, clip=True):
        """Predicts x0 from xt using the estimated noise."""
        # Get alpha values
        sqrt_recip_alphas = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t)
        sqrt_recipm1_alphas = self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        
        # Predict x0
        x0_pred = sqrt_recip_alphas * x_t - sqrt_recipm1_alphas * noise
        
        # Optionally clip x0 prediction
        if clip:
            x0_pred = torch.clamp(x0_pred, min=-1.0, max=1.0)
        
        return x0_pred

    def extract(self, a, t, x_t):
        """Extract parameters for a specific timestep and reshape for broadcasting."""
        batch_size = x_t.shape[0]
        out = a[t].to(x_t.device)
        return out.view(batch_size, 1, 1, 1)

    @torch.no_grad()
    def sampling(self, model, initial_noise, condition, seg_map=None, save_all_steps=False):
        """
        Sampling from the model using learned mean and variance.
        """
        image = initial_noise.to(self.device)
        images = []

        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # Create tensor of current timestep
            ts = torch.full((image.shape[0],), timestep, dtype=torch.long, device=self.device)
            
            # Get model prediction
            predicted_noise, v = model(image, ts, condition, seg_map)
            
            # Ensure v has consistent shape and is properly bounded
            if v.ndim == 2:  # If v is [B, D]
                v = v.mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
            elif v.ndim == 4 and v.shape[1] == 1:  # If v is [B, 1, H, W]
                pass  # Keep as is
            else:
                # Handle any other shape - average to get one value per batch item
                v = v.view(v.shape[0], -1).mean(dim=1).view(-1, 1, 1, 1)
            
            # Extract parameters for this timestep
            beta_t = self.extract(self.betas, ts, image)
            alpha_t = self.extract(self.alphas, ts, image)
            alpha_bar_t = self.extract(self.alphas_cumprod, ts, image)
            alpha_bar_prev_t = self.extract(self.alphas_cumprod_prev, ts, image)
            
            # Compute posterior variance
            # β̃_t = (1-α̅_(t-1))/(1-α̅_t) * β_t
            posterior_variance = (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * beta_t
            
            # Compute log variances (for interpolation)
            log_beta_t = torch.log(beta_t)
            log_posterior = torch.log(posterior_variance)
            
            # Interpolate log variance with v as the weight
            # Less aggressive clamping for numerical stability
            log_variance = v * log_beta_t + (1.0 - v) * log_posterior
            log_variance = log_variance.clamp(min=-10.0, max=4.0)
            variance = torch.exp(log_variance)
            
            # Predict x0 from noisy x_t
            x0_pred = self.predict_x0(image, ts, predicted_noise, clip=True)
            
            # Compute posterior mean using the correct formula
            # μ_q(x_(t-1) | x_t, x_0) = √α̅_(t-1) * β_t / (1-α̅_t) * x_0 + √α_t * (1-α̅_(t-1)) / (1-α̅_t) * x_t
            posterior_mean_coef1 = (beta_t * torch.sqrt(alpha_bar_prev_t)) / (1.0 - alpha_bar_t)
            posterior_mean_coef2 = ((1.0 - alpha_bar_prev_t) * torch.sqrt(alpha_t)) / (1.0 - alpha_bar_t)
            posterior_mean = posterior_mean_coef1 * x0_pred + posterior_mean_coef2 * image
            
            # Sample x_(t-1) from the posterior distribution
            noise = torch.zeros_like(image) if timestep == 0 else torch.randn_like(image)
            image = posterior_mean + torch.sqrt(variance) * noise
            
            # Debug prints for key statistics
            print(f"[Timestep {timestep}] Noise pred stats: mean={predicted_noise.mean().item():.4f}, std={predicted_noise.std().item():.4f}")
            print(f"[Timestep {timestep}] v stats: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
            print(f"[Timestep {timestep}] x0_pred stats: min={x0_pred.min().item():.4f}, max={x0_pred.max().item():.4f}")
            print(f"[Timestep {timestep}] variance: min={variance.min().item():.6f}, max={variance.max().item():.6f}")
            
            if save_all_steps:
                images.append(image.cpu())

        print(f"Final sampled output: min={image.min().item()}, max={image.max().item()}")
        return images if save_all_steps else image