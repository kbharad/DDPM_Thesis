# stats_pipeline.py - Optimized Dataset Statistics Computation

import os
import numpy as np
import pandas as pd
import logging
from train import training_config


def compute_dataset_statistics(cleaned_files):
    """
    Computes mean, std, min, and max for each variable in the dataset.
    """
    print("🔹 Computing dataset statistics...")
    
    aggregated_data = {}
    
    for file_path in cleaned_files: # Iterate over all cleaned NPZ files
        data = np.load(file_path) # Load npz files
        
        # data.files -> list of variables in the NPZ file
        for array_name in data.files:  # Extracts variable names, data.files ->Ex. 'schlieren'
            array = data[array_name]
            if training_config.crop_half == "True" and array.ndim >= 2:
                array = array[:, :array.shape[1] // 2]
            array = array.flatten()
            
            if array_name not in aggregated_data:
                aggregated_data[array_name] = []
            aggregated_data[array_name].extend(array)
    
    print("✅ Data aggregation complete. Calculating statistics...")
    
    # Compute statistics in one pass
    stats = [{
        "Array Name": variable,
        "Mean": np.mean(values),
        "Std Dev": np.std(values),
        "Min": np.min(values),
        "Max": np.max(values)
    } for variable, values in aggregated_data.items()]
    
    df = pd.DataFrame(stats)
    print(" Dataset statistics computed successfully!")
    return df