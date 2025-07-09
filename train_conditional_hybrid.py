import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
###########
#from model.unet_cond_binary import UNet # -> standard; works great
#from model.unet_flexible import UNet
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
            timesteps = torch.randint(0, diffusion_pipeline.num_timesteps, (original_images.shape[0],), device=device).long()

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
