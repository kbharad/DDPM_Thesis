# train.py - Main Training Script

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
###########
#from model.unet_cond_binary import UNet # -> standard; works great
from model.unet_multiple_injection import UNet # -> multiple injection
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


seed = 42
random.seed(seed)

# Set random seed for NumPy
np.random.seed(seed)

# Set random seed for PyTorch (both CPU and GPU)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # If using multiple GPUs



def main():
    print("schlieren conditioned dDPM, with edge based noising")
    logging.info("Training script started.")
    print(f" Conditioning Script Running: Conditioning on {training_config.condition_variable}")
    
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

    logging.info(f"✅ Results directory: {results_dir}")
    logging.info(f"✅ Checkpoints saved in: {checkpoints_dir}")
    logging.info(f"✅ Images saved in: {image_dir}")
    logging.info(f"✅ Stats saved in: {stats_dir}")
    logging.info(f"✅ Cleaned data saved in: {cleaned_data_dir}")
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
            #

            # Forward diffusion process
            noisy_images, noise = diffusion_pipeline.forward_diffusion(original_images, timesteps)
            noise_pred = model(noisy_images, timesteps, labels, cond_map=condition_frame)

            # calculate the loss, back propogate and update weights
            loss = F.mse_loss(noise_pred, noise)

            mean_loss += (loss.detach().item() - mean_loss) / (step + 1)
            
            optimizer.zero_grad()
            loss.backward()
            # prevent exploding gradients by clipping gradient at 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()            

            progress_bar.update(1)
            progress_bar.set_postfix(loss=mean_loss, lr=lr_scheduler.get_last_lr()[0], step=global_step)
            global_step += 1
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
            if epoch % 20== 0:
                df = pd.read_csv(loss_csv_path)  # Read training loss CSV

                plt.figure(figsize=(8, 5))
                plt.plot(df["Epoch"], df["Mean Training Loss"], marker='o', linestyle='-')

                plt.xlabel("Epoch")
                plt.ylabel("Mean Training Loss")
                plt.title("Training Loss Progress")
                plt.grid(True)

                plot_path = os.path.join(results_dir, "loss_plot.png")
                plt.savefig(plot_path, dpi=300)  # Save the updated plot
                plt.close()  # Prevent memory issues
                logging.info(f"Loss plot updated: {plot_path}")

        except Exception as e:
            logging.error(f" Error updating loss plot: {e}")


        # Evaluation after every save_image_epoch from train.py
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
