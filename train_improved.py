import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from train import training_config
from utils import save_checkpoint, setup_experiment
from torch.utils.data import DataLoader
import os
import logging
import csv
from data_setup import load_cfd_data
import matplotlib.pyplot as plt
import pandas as pd
#######
from model.ddpm_improved import DDPMPipeline
from evaluate_model_improved import evaluate_model
from model.unet_improved import UNet
########


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)


def compute_hybrid_loss(predicted_noise, v, target_noise, x_t, x_0,
                      log_beta, log_beta_tilde, alpha_t, alpha_bar_t, alpha_bar_prev_t,
                      lambda_vlb=0.001):  # Reduced weight for variational loss
    
    # 1. Simple MSE loss for noise prediction
    loss_simple = F.mse_loss(predicted_noise, target_noise)
    
    # 2. Calculate predicted log variance using interpolation
    log_variance = v * log_beta + (1 - v) * log_beta_tilde
    variance = torch.exp(log_variance.clamp(min=-10.0, max=4.0))  # Less aggressive clamping
    
    # 3. Calculate true posterior mean-> refer equation 11 of paper (praful, nichol)
    # μ_q(x_(t-1) | x_t, x_0) = √α̅_(t-1) * β_t / (1-α̅_t) * x_0 + √α_t * (1-α̅_(t-1)) / (1-α̅_t) * x_t
    coef_x0 = (torch.sqrt(alpha_bar_prev_t) * (1 - alpha_t)) / (1 - alpha_bar_t + 1e-8)
    coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev_t)) / (1 - alpha_bar_t + 1e-8)
    true_posterior_mean = coef_x0 * x_0 + coef_xt * x_t
    
    # 4. Calculate predicted posterior mean
    # Use x_0 prediction from the model's noise prediction -> x0 equation as function of xt, alpha t
    x0_from_noise = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
    x0_from_noise = x0_from_noise.detach()  # Detach to prevent gradients through this path
    
    # Re-compute posterior mean using the x_0 prediction
    pred_posterior_mean = coef_x0 * x0_from_noise + coef_xt * x_t
    
    # 5. Compute KL divergence between true and predicted posteriors
    # Both are Gaussians, so KL has  closed form
    true_posterior_variance = (1.0 - alpha_bar_prev_t) / (1.0 - alpha_bar_t) * alpha_t  # β̃_t
    
    # KL between two Gaussians: log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
    kl_div = 0.5 * (
    torch.log(variance + 1e-8) - torch.log(true_posterior_variance + 1e-8) + 
    (true_posterior_variance + (true_posterior_mean - pred_posterior_mean).pow(2)) / (variance + 1e-8) - 
    1.0
)
    
    loss_vlb = kl_div.mean()
    
    # 6. Combined loss with reduced VLB weight
    total_loss = loss_simple + lambda_vlb * loss_vlb
    
    return total_loss



def main():
    logging.info("Training script started.")
    print("Improved DDPM Model")

    # Ensure device-agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_config.device = device

    logging.info(f"Using device: {device}")
    
    #########################
    # Setup results directory using training_config.base_path
    results_dir = training_config.base_path
    os.makedirs(results_dir, exist_ok=True)
    
    image_dir = os.path.join(results_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    stats_dir = training_config.stats_folder  # Ensure stats directory exists
    os.makedirs(stats_dir, exist_ok=True)

    cleaned_data_dir = training_config.cleaned_output_path  # Ensure cleaned data directory exists
    os.makedirs(cleaned_data_dir, exist_ok=True)

    logging.info(f" Results directory: {results_dir}")
    logging.info(f" Checkpoints saved in: {checkpoints_dir}")
    logging.info(f" Images saved in: {image_dir}")
    logging.info(f" Stats saved in: {stats_dir}")
    logging.info(f" Cleaned data saved in: {cleaned_data_dir}")
    ###########################

    ###########################

    # Load preprocessed , dataloaders 
    train_dataloader, test_dataloader, stats_dict = load_cfd_data()
    print(f"Stats Dict: {stats_dict}")

    logging.info(f"Dataloaders created)")

    # Initialize Model
    model = UNet(input_channels=training_config.input_channels, num_conditions=3, condition_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader.dataset) * training_config.num_epochs, eta_min=1e-9)
    logging.info("Model initialized.")
    

    # Load checkpoint if resuming
    if training_config.resume:
        checkpoint = torch.load(training_config.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        training_config.start_epoch = checkpoint['epoch'] + 1

    diffusion_pipeline = DDPMPipeline(num_timesteps= training_config.diffusion_timesteps)
    global_step = training_config.start_epoch * len(train_dataloader.dataset)
    csv_filename = os.path.join(results_dir, "errors_epochs.csv")
    #################
    # Write CSV Header
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["SampleIndex", "MSE", "SSIM", "Epoch"]) 

    #################################################################################
    # Training loop
    logging.info("Starting training loop.....")
    loss_csv_path = os.path.join(results_dir, "training_loss.csv")
    file_exists = os.path.isfile(loss_csv_path)



    for epoch in range(training_config.start_epoch, training_config.num_epochs):
        print(f"Epoch {epoch}")

        progress_bar = tqdm(total=len(train_dataloader.dataset), desc=f"Epoch {epoch}")
        mean_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # each batch has Schlieren, label, and targets
            if step % 10 == 0:
                logging.info(f"Batch {step}/{len(train_dataloader)} - Loss: {mean_loss:.6f}")

            # (remember device agnostic code for debug)
            condition_frame = batch['schlieren'].to(device)
            labels = batch['label'].to(device)
            original_images = batch['targets'].to(device)

            # Sample random timesteps
            # each image gets unique t
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (original_images.shape[0],), device=device).long()

            # Forward diffusion process
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)

            #For Debugging
            print(f"Epoch {epoch} | Step {step} | Noised Images -> min: {noisy_images.min().item():.6f}, max: {noisy_images.max().item():.6f}, mean: {noisy_images.mean().item():.6f}, var: {noisy_images.var().item():.6f}")
            print(f"Epoch {epoch} | Step {step} | Noise Added  -> min: {noise.min().item():.6f}, max: {noise.max().item():.6f}, mean: {noise.mean().item():.6f}, var: {noise.var().item():.6f}")




            #  U-Net now returns both noise and log-variance
            predicted_noise, v = model(noisy_images, timesteps, labels, cond_map=condition_frame)


            log_beta = torch.log(diffusion_pipeline.betas[timesteps])
            log_beta_tilde = torch.log(diffusion_pipeline.alphas_cumprod_prev[timesteps])
            alpha_t = diffusion_pipeline.alphas[timesteps]
            alpha_bar_t = diffusion_pipeline.alphas_cumprod[timesteps]
            alpha_bar_prev_t = diffusion_pipeline.alphas_cumprod_prev[timesteps]

            #Reshape for broadcasting
            log_beta = log_beta.view(-1, 1, 1, 1)
            log_beta_tilde = log_beta_tilde.view(-1, 1, 1, 1)
            alpha_t = alpha_t.view(-1, 1, 1, 1)
            alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
            alpha_bar_prev_t = alpha_bar_prev_t.view(-1, 1, 1, 1)


            loss = compute_hybrid_loss(predicted_noise, v, noise, noisy_images, original_images,
                                    log_beta, log_beta_tilde, alpha_t, alpha_bar_t, alpha_bar_prev_t)


            mean_loss += (loss.detach().item() - mean_loss) / (step + 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()


            progress_bar.update(1)
            progress_bar.set_postfix(loss=mean_loss, lr=lr_scheduler.get_last_lr()[0], step=global_step)
            global_step += 1
        
        # Learning rate decay
        lr_scheduler.step()        
        logging.info(f" Epoch {epoch + 1} completed. Average loss: {mean_loss:.6f}")
        
        # logging the tranining loss (mean per epoch)
        
        with open(loss_csv_path, "a") as f:
            csv_writer = csv.writer(f)
            if not file_exists:
                csv_writer.writerow(["Epoch", "Mean Training Loss"])
            csv_writer.writerow([epoch + 1, f"{mean_loss:.6f}"])


        # curve of training loss to monitor 
        
        try:
            if epoch % 20 == 0:
                df = pd.read_csv(loss_csv_path)
                plt.figure(figsize=(8, 5))
                plt.plot(df["Epoch"], df["Mean Training Loss"], marker='o')
                plt.xlabel("Epoch")
                plt.ylabel("Mean Training Loss")
                plt.title("Training Loss Progress")
                plt.grid(True)
                plt.savefig(os.path.join(results_dir, "loss_plot.png"), dpi=300)
                plt.close()
                logging.info("✅ Loss plot saved.")
        except Exception as e:
            logging.warning(f"⚠️ Error plotting training curve: {e}")

        # Evaluation after every save_image_epoch from train.py
        #if (epoch + 1) % training_config.save_image_epochs == 0:
        if (epoch == 1) or ((epoch + 1) % training_config.save_image_epochs == 0):
            model.eval()
            evaluate_model(model, test_dataloader, diffusion_pipeline, training_config, epoch, csv_filename, image_dir)

        
        # Save model checkpoint
        if (epoch + 1) % training_config.save_model_epochs == 0:
            save_checkpoint(model, optimizer, lr_scheduler, training_config, epoch, checkpoints_dir)
            logging.info(f" Model checkpoint saved at epoch {epoch + 1}")


    # Save final model checkpoint
    save_checkpoint(model, optimizer, lr_scheduler, training_config, training_config.num_epochs, checkpoints_dir)
    logging.info(f" Final model checkpoint saved at epoch {training_config.num_epochs}")
    logging.info("Training complete!!!!")
if __name__ == "__main__":
    main()
