import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
###########
from model.unet_multiple_injection import UNet #combining hybrid loss and multiple injection
###########
from model.ddpm_version2 import DDPMPipeline
#from model.ddpm_edge_based import DDPMPipeline
from train import training_config
from utils import save_checkpoint, setup_experiment
from evaluate_model import evaluate_model
from torch.utils.data import DataLoader
import os
import logging
import csv

###########
from data_setup import load_cfd_data
###########

import matplotlib.pyplot as plt
import pandas as pd
#Trying some logging code since no logs were visible, so will put multiple logs throughout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
import random
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure as ssim



seed = training_config.seed
random.seed(seed)

# Set random seed for NumPy
np.random.seed(seed)

# Set random seed for PyTorch (both CPU and GPU)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs



def main():
    logging.info("Training script started.")
    print(f" Conditioning Script Running: Conditioning on {training_config.condition_variable}")
    
    training_config.mse_weight = 0.8  # Weight for MSE loss component
    training_config.ssim_weight = 0.2  # Weight for SSIM loss component
    logging.info(f"Using hybrid loss with MSE weight: {training_config.mse_weight}, SSIM weight: {training_config.ssim_weight}")

    

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
    sampling_plot_dir = os.path.join(results_dir, "sampling_probability")
    os.makedirs(sampling_plot_dir, exist_ok=True)
    sampling_stats_path = os.path.join(results_dir, "sampling_statistics.csv")
    with open(sampling_stats_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Epoch', 'Timestep', 'Average_Loss', 'Sampling_Weight'])

    ###########################

    ###########################

    # Load preprocessed , dataloaders 
    train_dataloader, test_dataloader, stats_dict = load_cfd_data()

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
    num_timesteps = diffusion_pipeline.num_timesteps

    # Non-uniform timestep sampling setup
    timestep_losses = torch.zeros(num_timesteps, device=device)
    timestep_counts = torch.zeros(num_timesteps, device=device)
    sampling_weights = torch.ones(num_timesteps, device=device) / num_timesteps
    use_weighted_sampling = False
    warmup_epochs = 5
    sampling_temperature = 4.0

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
        progress_bar = tqdm(total=len(train_dataloader.dataset), desc=f"Epoch {epoch}")
        mean_loss = 0
        mean_mse_loss = 0
        mean_ssim_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # each batch has Schlieren, label, and targets
            if step % 10 == 0:
                logging.info(f"Batch {step}/{len(train_dataloader)} - Loss: {mean_loss:.6f}")

            # (remember device agnostic code for debug)
            condition_frame = batch[training_config.condition_variable].to(device)
            labels = batch['label'].to(device)
            original_images = batch['targets'].to(device)
            if epoch == 0:
                print(f"Original Image Shape: {original_images.shape}")
                print(f"Condition Frame Shape: {condition_frame.shape}")
                print(f"Labels Shape: {labels.shape}")
            # Sample random timesteps
            # each image gets unique t
            batch_size = original_images.shape[0]
            if use_weighted_sampling:
                timestep_indices = torch.multinomial(sampling_weights, batch_size, replacement=True)
                timesteps = timestep_indices.to(device)
            else:
                timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).long()

            # Forward diffusion process
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noise_pred = model(noisy_images, timesteps, labels, cond_map=condition_frame)

            # Calculate MSE loss on noise prediction (your original loss)
            mse_loss = F.mse_loss(noise_pred, noise)
            
            # Calculate x0 prediction using your existing predict_x0 method
            x0_pred = diffusion_pipeline.predict_x0(noisy_images, timesteps, noise_pred)

            # Calculate SSIM loss using torchmetrics
            
            # use 1 - SSIM as the loss term to make it an optimization problem
            ssim_value = ssim(x0_pred, original_images, data_range=2.0)  
            ssim_loss = 1.0 - ssim_value

            # Combine the losses with their weights
            loss = training_config.mse_weight * mse_loss + training_config.ssim_weight * ssim_loss

            # Update running averages for all loss components
            mean_mse_loss += (mse_loss.detach().item() - mean_mse_loss) / (step + 1)
            mean_ssim_loss += (ssim_loss.detach().item() - mean_ssim_loss) / (step + 1)
            mean_loss += (loss.detach().item() - mean_loss) / (step + 1)
            
            optimizer.zero_grad()
            loss.backward()
            # prevent exploding gradients by clipping gradient at 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # Per-image MSE loss for timestep stats
            loss_per_image = F.mse_loss(noise_pred, noise, reduction='none')
            loss_per_image = loss_per_image.mean(dim=(1, 2, 3))

            with torch.no_grad():
                for i in range(batch_size):
                    t = timesteps[i].item()
                    current_loss = loss_per_image[i].item()
                    if timestep_counts[t] == 0:
                        timestep_losses[t] = current_loss
                    else:
                        timestep_losses[t] = (timestep_counts[t] * timestep_losses[t] + current_loss) / (timestep_counts[t] + 1)
                    timestep_counts[t] += 1


            progress_bar.update(1)
            progress_bar.set_postfix(
                total=mean_loss, 
                mse=mean_mse_loss, 
                ssim=mean_ssim_loss, 
                lr=lr_scheduler.get_last_lr()[0], 
                step=global_step
            )
            global_step += 1
        lr_scheduler.step()
        if epoch >= warmup_epochs and torch.sum(timestep_counts > 0) > 0.9 * num_timesteps:
            valid_timesteps = timestep_counts > 0
            smoothed_losses = timestep_losses.clone()
            
            for t in range(num_timesteps):
                if timestep_counts[t] == 0:
                    left_idx = t - 1
                    while left_idx >= 0 and timestep_counts[left_idx] == 0:
                        left_idx -= 1
                    right_idx = t + 1
                    while right_idx < num_timesteps and timestep_counts[right_idx] == 0:
                        right_idx += 1
                    if 0 <= left_idx and right_idx < num_timesteps:
                        smoothed_losses[t] = (smoothed_losses[left_idx] + smoothed_losses[right_idx]) / 2
                    elif 0 <= left_idx:
                        smoothed_losses[t] = smoothed_losses[left_idx]
                    elif right_idx < num_timesteps:
                        smoothed_losses[t] = smoothed_losses[right_idx]
                    else:
                        smoothed_losses[t] = timestep_losses[valid_timesteps].mean()
            
            powered_losses = smoothed_losses ** (1 / sampling_temperature)
            new_weights = powered_losses / powered_losses.sum()
            new_weights = torch.clamp(new_weights, min=1e-4)
            new_weights = new_weights / new_weights.sum()
            sampling_weights = new_weights
            use_weighted_sampling = True
        # Save sampling statistics to CSV
        with open(sampling_stats_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for t in range(num_timesteps):
                csv_writer.writerow([
                    epoch, 
                    t, 
                    f"{timestep_losses[t].item():.6f}", 
                    f"{sampling_weights[t].item():.6f}"
                ])

        # Visualize sampling distribution periodically
        if epoch % 5 == 0 or epoch == training_config.num_epochs - 1:
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 1)
            plt.bar(np.arange(num_timesteps), sampling_weights.cpu().numpy(), width=1.0)
            plt.title(f"Timestep Sampling Distribution (Epoch {epoch+1})")
            plt.ylabel("Sampling Probability")
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.subplot(2, 1, 2)
            valid_timesteps = timestep_counts > 0
            valid_indices = np.arange(num_timesteps)[valid_timesteps.cpu().numpy()]
            valid_losses = timestep_losses[valid_timesteps].cpu().numpy()

            plt.bar(valid_indices, valid_losses, width=1.0)
            plt.title(f"Average Loss per Timestep (Epoch {epoch+1})")
            plt.xlabel("Timestep")
            plt.ylabel("Average Loss")
            plt.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.savefig(os.path.join(sampling_plot_dir, f"sampling_distribution_epoch_{epoch+1}.png"))
            plt.close()


        logging.info(f" Epoch {epoch + 1} completed. Total loss: {mean_loss:.6f}, MSE: {mean_mse_loss:.6f}, SSIM: {mean_ssim_loss:.6f}")
        
        # If you're logging to CSV, update to include both loss components
        with open(loss_csv_path, "a") as f:
            csv_writer = csv.writer(f)
            if epoch == 0:  # Write header if first epoch
                csv_writer.writerow(["Epoch", "Total Loss", "MSE Loss", "SSIM Loss"])
            csv_writer.writerow([epoch + 1, f"{mean_loss:.6f}", f"{mean_mse_loss:.6f}", f"{mean_ssim_loss:.6f}"])


        # curve of training loss to monitor 
        
        try:
            if epoch % 10 == 0:  # Update plot every 10 epochs
                # Read the CSV file with all loss data
                df = pd.read_csv(loss_csv_path)
                
                # Create a figure with appropriate size
                plt.figure(figsize=(12, 7))
                
                # Plot all three loss components on the same graph
                plt.plot(df["Epoch"], df["Total Loss"], marker='o', linestyle='-', label='Total Loss', color='blue')
                plt.plot(df["Epoch"], df["MSE Loss"], marker='s', linestyle='--', label='MSE Loss', color='red')
                plt.plot(df["Epoch"], df["SSIM Loss"], marker='^', linestyle='-.', label='SSIM Loss', color='green')
                
                # Add labels and title
                plt.xlabel("Epoch", fontsize=12)
                plt.ylabel("Loss Value", fontsize=12)
                plt.title("Training Loss Components Over Time", fontsize=14)
                
                # Add grid and legend
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                
                # Ensure y-axis starts at 0 for better visual comparison
                plt.ylim(bottom=0)
                
                # Add some padding around the plot
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(results_dir, "loss_components_plot.png")
                plt.savefig(plot_path, dpi=300)
                plt.close()  # Close to prevent memory issues
                
                logging.info(f"Loss components plot updated: {plot_path}")
        except Exception as e:
            logging.error(f"Error updating loss plot: {e}")
            # Print the full exception traceback for debugging
            import traceback
            logging.error(traceback.format_exc())


        # Evaluation after every save_image_epoch from train.py
        if ((epoch + 1) % training_config.save_image_epochs == 0):
            model.eval()
            evaluate_model(model, test_dataloader, diffusion_pipeline, training_config, epoch, csv_filename, image_dir)

        
        # Save model checkpoint
        if (epoch + 1) % training_config.save_model_epochs == 0 or (epoch+1)== 150:
            save_checkpoint(model, optimizer, lr_scheduler, training_config, epoch, checkpoints_dir)
            logging.info(f" Model checkpoint saved at epoch {epoch + 1}")


    # Save final model checkpoint
    save_checkpoint(model, optimizer, lr_scheduler, training_config, training_config.num_epochs, checkpoints_dir)
    logging.info(f" Final model checkpoint saved at epoch {training_config.num_epochs}")
    logging.info("Training complete!!!!")
if __name__ == "__main__":
    main()
