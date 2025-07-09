# data_setup.py - Optimized Data Preprocessing
from torchvision import transforms
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
#from data_preprocessing.OutliersRemoval import clean_dataset_path
from data_preprocessing.StatsPipeline import compute_dataset_statistics
from data_preprocessing.BicubicResampling import ResamplingCFDDataset
from train import training_config
from data_preprocessing.DataLoading import create_dataloaders
#new preprocessing
from data_preprocessing.outlier_treatmentv2 import process_npz_files
import random
SEED = training_config.seed  # Use the same seed as in DataLoading.py

# Set all seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # For all GPU devices
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # For reproducibility






def load_cfd_data():
    """
    Loads and preprocesses CFD dataset: removes outliers, computes statistics, returns dataloader
    """
    print(" Loading and preprocessing CFD dataset...")

    SEED = training_config.seed  # Use the same seed as in DataLoading.py

    # Set all seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # For all GPU devices
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # For reproducibility

    # Paths
    dataset_path = training_config.dataset_path  # Raw data path
    output_dir = training_config.processed_npz_path  # Cleaned data output path
    
    # Clean dataset (remove outliers)
    print(" Starting outlier cleaning process...")
    process_npz_files(dataset_path, output_dir)  # New outlier handling function
    print(f" Outlier cleaning completed. Cleaned data stored in: {output_dir}")
    
    # Prepare cleaned dataset path list
    clean_dataset_path_arr = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npz')]
    print(f" Outlier removal complete. Remaining samples: {len(clean_dataset_path_arr)}")
    
    # Compute dataset statistics
    stats_df = compute_dataset_statistics(clean_dataset_path_arr)
    stats_dict = {row['Array Name']: {
        "mean": row['Mean'], "std": row['Std Dev'], "min": row['Min'], "max": row['Max']
    } for _, row in stats_df.iterrows()}
    print(" Dataset statistics computed.")

    # Define target variables (modify if needed)
    target_variables = training_config.target_variables  # Example: ['velocity_magnitude', 'density', 'mach_number']
    
    # Create dataset splits
    train_dataset, test_dataset = create_dataloaders(output_dir, clean_dataset_path_arr, target_variables, stats_dict, training_config.train_batch_size, seed = SEED)
    
    # Apply bicubic resampling
    train_resampled_dataset = ResamplingCFDDataset(train_dataset)
    test_resampled_dataset = ResamplingCFDDataset(test_dataset)
    print("Resampling Done")

    
    # Convert to PyTorch DataLoaders
    train_dataloader = DataLoader(train_resampled_dataset, batch_size=training_config.train_batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_resampled_dataset, batch_size=training_config.eval_batch_size, shuffle=False, num_workers=2)
    
    print(" Data preprocessing complete. Ready for training!")
    return train_dataloader, test_dataloader, stats_dict
