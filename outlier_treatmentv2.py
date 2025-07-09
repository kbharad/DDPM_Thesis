import os
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.stats import zscore
import csv
import logging
import shutil


def detect_outlier_region(arr, threshold=4.0, gradient_threshold=1):
    """
    Detects outlier regions by identifying extreme values using Z-score
    and checking if neighboring pixels have a high gradient.
    """
    z_scores = np.abs(zscore(arr)) #z score for every pixel in image
    outlier_indices = np.argwhere(z_scores > threshold)
    affected_pixels = set()

    for x, y in outlier_indices:
        affected_pixels.add((x, y))
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
            xn, yn = x + dx, y + dy
            if 0 <= xn < arr.shape[0] and 0 <= yn < arr.shape[1]:
                if abs(arr[xn, yn] - arr[x, y]) > gradient_threshold:
                    affected_pixels.add((xn, yn))
    return list(affected_pixels) #list of tuples of affected indices

def replace_with_local_mean(arr, x, y):
    """
    Replace the outlier pixel (x, y) with the mean of its surrounding valid pixels.
    """
    neighbors = []
    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (-1,1), (1,-1)]:
        xn, yn = x + dx, y + dy
        if 0 <= xn < arr.shape[0] and 0 <= yn < arr.shape[1]:
            neighbors.append(arr[xn, yn])
    if neighbors:
        return np.mean(neighbors)
    else:
        return arr[x, y]  # Fallback to original value if no neighbors


def clean_outliers(arr):
    """
    Detects and replaces outliers using the Z-score and nearby pixels approach.
    """
    affected_pixels = detect_outlier_region(arr)
    arr_cleaned = arr.copy()
    for x, y in affected_pixels:
        arr_cleaned[x, y] = replace_with_local_mean(arr, x, y)
    return arr_cleaned



def process_npz_files(input_path, output_path):
    """
    Processes all .npz files in the specified input directory, cleans outliers, and saves cleaned files.
    """
    # Ensure the output directory is fresh
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    for file_name in os.listdir(input_path):
        if file_name.endswith('.npz'):
            file_path = os.path.join(input_path, file_name)
            data = np.load(file_path)
            data_dict = {}
            for key in data.files:
                if key == "schlieren":
                    # Directly copy Schlieren data without cleaning
                    data_dict[key] = data[key]
                else:
                    # Clean other variables
                    data_dict[key] = clean_outliers(data[key])
            output_file_path = os.path.join(output_path, file_name)
            np.savez(output_file_path, **data_dict)


