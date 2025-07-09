# bicubic_resampling.py - Optimized Bicubic Resampling for CFD Data
# note to self -> for halfed images, bicubic is too blur

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from train import training_config

class ResamplingCFDDataset(Dataset):
    """
    Resamples or crops CFD dataset images.
    If training_config.crop_half is True, the image is cropped to its left half and then
    bicubic interpolation is applied to smooth and resize it to the desired target size.
    Otherwise, bicubic interpolation is applied directly to the full image.
    """
    def __init__(self, tensor_data, target_size=(80, 256)):
        self.data = tensor_data  
        # target_size remains the final desired size (e.g. (80,128) if cropping is enabled)
        self.target_size = target_size
        if training_config.crop_half == "True":
            self.target_size = (target_size[0], target_size[1] // 2)
            self.resample_mode = "bilinear"
        elif training_config.crop_half == "False":
            self.target_size = target_size
            self.resample_mode = "bicubic"
        #self.resample_mode = resample_mode
        self.condition_variable = training_config.condition_variable

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if training_config.crop_half == "True":
            # Use clone().detach().float() to safely copy if sample[...] is already a tensor.
            condition_tensor = sample[self.condition_variable].clone().detach().float() # since i was getting some warnings about copying tensors 
            targets_tensor = sample['targets'].clone().detach().float()

            # Ensure the tensors have 4 dimensions (N, C, H, W). If they are 3D, add a batch dimension.
            if condition_tensor.dim() == 3:
                condition_tensor = condition_tensor.unsqueeze(0)  # shape: (1, H, W)
            if targets_tensor.dim() == 3:
                targets_tensor = targets_tensor.unsqueeze(0)

            # Crop the left half along the width
            cropped_condition = condition_tensor[:, :, :, :condition_tensor.shape[-1] // 2]
            cropped_targets = targets_tensor[:, :, :, :targets_tensor.shape[-1] // 2]

            # Apply bicubic interpolation to the cropped tensors for smoothing and resizing to target_size
            condition_resampled = F.interpolate(cropped_condition, size=self.target_size, mode=self.resample_mode, align_corners=False)
            targets_resampled = F.interpolate(cropped_targets, size=self.target_size, mode=self.resample_mode, align_corners=False)

            # Remove the added batch dimension for consistency with downstream processing
            condition_resampled = condition_resampled.squeeze(0)
            targets_resampled = targets_resampled.squeeze(0)
        elif training_config.crop_half == "False":
            # Directly resample the full image using bicubic interpolation
            condition_resampled = F.interpolate(
                torch.tensor(sample[self.condition_variable], dtype=torch.float32).unsqueeze(0),
                size=self.target_size,
                mode=self.resample_mode,
                align_corners=False
            ).squeeze(0)
            targets_resampled = F.interpolate(
                torch.tensor(sample['targets'], dtype=torch.float32).unsqueeze(0),
                size=self.target_size,
                mode=self.resample_mode,
                align_corners=False
            ).squeeze(0)

        return {
            self.condition_variable: condition_resampled,
            "targets": targets_resampled,
            "label": sample["label"]
        }

