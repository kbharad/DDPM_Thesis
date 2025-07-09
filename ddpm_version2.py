import torch
import numpy as np
import math
from tqdm import tqdm
from typing import Tuple
# best one so far

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    return np.array(betas, dtype=np.float64)

def sigmoid_schedule(beta_start, beta_end, timesteps, start=-6, end=6):
    x = np.linspace(start, end, timesteps)
    sig = 1 / (1 + np.exp(-x))
    sig = (sig - sig.min()) / (sig.max() - sig.min())
    return sig * (beta_end - beta_start) + beta_start

def broadcast(values, broadcast_to):
    values = values.flatten()
    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)
    return values

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
        elif schedule_name == "sigmoid":
            self.betas = torch.tensor(
        sigmoid_schedule(1e-4, 0.02, num_timesteps),
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

        # Both timesteps and self.alphas_cumprod are on the same device
        alpha_cumprod_t = self.alphas_cumprod[timesteps]  

        # alpha_cumprod_t is moved to images' device, reshape it for broadcasting (shape errors)
        alpha_cumprod_t = alpha_cumprod_t.to(images.device).view(batch_size, 1, 1, 1)

        gaussian_noise = torch.randn_like(images, device=images.device)

        # Add noise to the images 
        # standard forward diffusion formula xt = ..

        noised_image = torch.sqrt(alpha_cumprod_t) * images + torch.sqrt(1 - alpha_cumprod_t) * gaussian_noise

        return noised_image, gaussian_noise

    
    def predict_x0(self, x_t, t, noise, clip=True):
        """Predicts x0 from xt using the estimated noise and clips it if needed.
        predicting x0 and using posterior mean is better than just subtracting noise, gives a good target for the model to learn"""
        x0_pred = self.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * noise
        
        if clip:
            x0_pred = torch.clamp(x0_pred, min=-1.0, max=1.0)  
        
        return x0_pred

    # try with clipping x0_pred, should guide it correctly, without clipping results are still bad
    #def predict_x0(self, x_t, t, noise):
    #    """Predicts x0 from xt using the estimated noise."""
    #    return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * noise

    def q_posterior(self, x_start, x_t, t):
        """Computes posterior mean for reverse diffusion."""
        coef1 = self.extract(self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod + 1e-6), t, x_t)
        coef2 = self.extract((1.0 - self.alphas_cumprod_prev) * torch.sqrt(1.0 - self.betas) / (1.0 - self.alphas_cumprod + 1e-6), t, x_t)

        # Refer to equation 7, DDPM paper: https://arxiv.org/pdf/2006.11239
        return coef1 * x_start + coef2 * x_t

    def extract(self, a, t, x_shape):
        """Extracts values for a specific timestep and reshapes for broadcasting."""
        batch_size = x_shape.shape[0]
        out = a[t].to(x_shape.device)  #  Ensure correct device
        return out.view(batch_size, 1, 1, 1)  # Reshape for broadcasting
        

    @torch.no_grad()
    def sampling(self, model, initial_noise, condition, seg_map=None, save_all_steps=False):
        """
        Performs the reverse diffusion process to sample an image from noise.
        """
        self.betas = self.betas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        image = initial_noise.to(self.device)  # Ensure noise starts on the correct device
        images = []

        condition = condition.to(self.device)  # condition data on correct device
        seg_map = seg_map.to(self.device) if seg_map is not None else None
        

        # Enable mixed precision to reduce memory usage
        scaler = torch.amp.GradScaler()



        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            # timestep information, broadcast to necessary shape
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=self.device)

            #with torch.amp.autocast(device_type="cuda"):
                # extra attention block causing memory issues, so trying this..
            #    predicted_noise = model(image, ts, condition, seg_map)
            #    x0_pred = self.predict_x0(image, ts, predicted_noise)
            #    posterior_mean = self.q_posterior(x0_pred, image, ts)

            # Predict noise using U-Net
            predicted_noise = model(image, ts, condition, seg_map)

            # Predict x0 using the estimated noise and also clipped from -1,1
            x0_pred = self.predict_x0(image, ts, predicted_noise)

            # Compute the posterior mean
            posterior_mean = self.q_posterior(x0_pred, image, ts)

            # Add variance noise (only if t > 0)
            noise = torch.randn_like(image, device=self.device) if timestep > 0 else 0

            # reparametrize equation; mean + sigma.Z; sigma = beta(t), z = fresh noise injected to prevent collapse to single image
            image = posterior_mean + noise * self.extract(torch.sqrt(self.betas), ts, image)


            del predicted_noise, x0_pred, posterior_mean, noise 
            torch.cuda.empty_cache() # 
            # Debugging print
            if timestep % 50 == 0:
                print(f"Image at Timestep {timestep} | min: {image.min().item():.6f}, max: {image.max().item():.6f}, mean: {image.mean().item():.6f}, var: {image.var().item():.6f}")

            if save_all_steps:
                images.append(image.cpu())

        print(f"Final sampled output: min={image.min().item()}, max={image.max().item()},mean={ image.mean().item()},var ={ image.var().item()}")
        return images if save_all_steps else image













