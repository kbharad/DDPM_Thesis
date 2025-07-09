import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import re
import logging
import matplotlib.pyplot as plt
from train import training_config


# Information about the dataset: 
# Pressure: Min = 5.00 bar, Max = 74.98 bar
# Temperature: Min = 274.01 K, Max = 673.91 K

class CFD_preprocessing(Dataset):
    def __init__(self, directory, cleaned_datafiles, target_variables, stats_dict, preprocessing=training_config.processing_data,condition_variable =training_config.condition_variable, scaling_min=-1, scaling_max=1):
        """
        Args:
            directory (str): npz folder filepath
        """
        self.directory = directory
        self.file_names = cleaned_datafiles
        self.condition_variable = condition_variable
        self.stats_dict = stats_dict

        self.preprocessing = preprocessing
        if self.preprocessing == "scaling" or self.preprocessing == "scaling_then_normalization" or self.preprocessing == "normalization_then_scaling":
            # this makes sure that the values are in numberical format, which might happen if they are called from command line
            self.scaling_min =  float(scaling_min)
            self.scaling_max =  float(scaling_max)
        self.target_variables = target_variables


        # crop_half 
        self.crop_half = training_config.crop_half


    # using for rescaling and normalization
    def preprocess_data(self, tensor, variable):
        if self.preprocessing == "normalization":
            mean = self.stats_dict[variable]["mean"]
            std = self.stats_dict[variable]["std"]
            # performing normalization
            tensor = (tensor - mean)/std
            logging.info(f"{self.preprocessing} done successfully ")
            
        if self.preprocessing == "scaling":
            min = self.stats_dict[variable]["min"]
            max = self.stats_dict[variable]["max"]
            # performing scaling
            tensor = self.scaling_min + (((tensor - min)*(self.scaling_max - self.scaling_min))/(max - min))
            logging.info(f"{self.preprocessing} done successfully ")
            
        if self.preprocessing == "scaling_then_normalization":
            """
            First, scale the data between `scaling_min` and `scaling_max`.
            Then, normalize using mean and standard deviation **computed from the scaled data**.
            avoid this..
            """
            min_val = self.stats_dict[variable]["min"]
            max_val = self.stats_dict[variable]["max"]
            # Step 1: Scaling
            tensor = self.scaling_min + (((tensor - min_val)*(self.scaling_max - self.scaling_min))/(max_val - min_val))
            
            # Step 2: Compute new mean & std from the scaled data
            new_mean = self.scaling_min + (((self.stats_dict[variable]["mean"] - min_val)*(self.scaling_max - self.scaling_min))/(max_val - min_val))
            new_std = (self.stats_dict[variable]["std"] * (self.scaling_max - self.scaling_min)) / (max_val - min_val)
            
            # Step 3: Normalize using the **new** mean & std
            tensor = (tensor - new_mean) / new_std
            logging.info(f"{self.preprocessing} done successfully")

        if self.preprocessing == "normalization_then_scaling":
            """
            First, normalize using mean and standard deviation.
            Then, scale the normalized data between `scaling_min` and `scaling_max`.
            This ensures output is strictly in (-1,1) 
            makes more sense.
            Generally DDPM Unet prefers inputs in [-1,1]
            """
            mean = self.stats_dict[variable]["mean"]
            std = self.stats_dict[variable]["std"]
            
            # Step 1: Normalize (Zero mean, unit variance)
            tensor = (tensor - mean) / std
            
            # Step 2: Scale the normalized values to [-1,1]
            min_norm = tensor.min()
            max_norm = tensor.max()
            tensor = self.scaling_min + (((tensor - min_norm) * (self.scaling_max - self.scaling_min)) / (max_norm - min_norm))
            
            logging.info(f"{self.preprocessing} done successfully")
        return tensor 
    
    

    def scale_labels(self, tensor, variable):
    # Pressure: Min = 5.00 bar, Max = 74.98 bar
    # Temperature: Min = 274.01 K, Max = 673.91 K
        if variable == "pressure":
            minimum = 5.0
            maximum = 74.98        
        if variable == "temperature":
            minimum = 274.01
            maximum = 673.91
        
        logging.info(f"Scaling label: {variable} to [-1,1]")
        return 2 * (tensor - minimum) / (maximum - minimum) - 1  # Scales to [-1,1]



    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Obtain the file path
        # file_path = os.path.join(self.directory, self.file_names[idx])
        file_path = self.file_names[idx]
        data = np.load(file_path, mmap_mode="r")

        # Convert Schlieren Image to Pytorch Tensor
        condition_data = self.preprocess_data(data[self.condition_variable], variable=self.condition_variable)
        if training_config.crop_half == True:
            print(f"Cropping data to half: {training_config.crop_half}")
            condition_data = condition_data[:, :128]


        # converting data into shape (num_samples, 1,H,W) --- (1, 1, H, W) - for single datapoint
        condition_tensor = torch.tensor(condition_data, dtype=torch.float32).unsqueeze(0)
        logging.info(f"The unprocessed shape of {self.condition_variable} is {condition_tensor.shape}")


        var_array = []
        # Extract data from the npz file
        for variable in self.target_variables:
            var_data = self.preprocess_data(data[variable], variable=variable)

            if training_config.crop_half == True:
                var_data = var_data[:, :128]  # Crop the width

            var_data = torch.tensor(var_data, dtype=torch.float32).unsqueeze(0)
            var_array.append(var_data)
        logging.info(f"The target variables(output of U-net) of generation are stored in var_array of shape {len(var_array)}")

        logging.info(f"Combining all the target variables so the target variables will of shape (1, num_of target_vars, H, W)")
        if len(self.target_variables)>1:
            targets = torch.cat(var_array, dim=0)
        else:
            targets = var_data

        # Extract materials, pressure, and temperature from the file name
        file_path = self.file_names[idx]
        file_name = os.path.basename(file_path)  # ✅ FIX: extract just the filename

        match = re.match(r'(.+)_press_(.+?)bar_temp_(.+?)K\.npz', file_name)
        try:
            if match:
                material = match.group(1)  # now safely gives 'argon' or 'nitrogen'
                pressure = float(match.group(2))
                temperature = float(match.group(3))
            else: 
                raise ValueError(f"File name does not match: {file_name}")  
        except Exception as e:
            logging.info(f"Unexpected Error Occured while extracting pressure and temperation is: {e}")
            
        # Create label tensor, argen ->0 and nitrogen->1
        material_dict = {'argon': 0, 'nitrogen': 1}
        material_label = material_dict.get(material, -1)   # if none of the gasses are present it will treturn -1 
        # label tensor is shape 3-> material, temperature, pressure
        label_tensor = torch.tensor([material_label, self.scale_labels(pressure, variable="pressure"), 
                                     self.scale_labels(temperature, variable="temperature")], 
                                    dtype=torch.float32)
        logging.info(f"Successfully extracted the pressure and temperature and converted it into tensors")


        return {self.condition_variable:condition_tensor, "targets":targets, "label":label_tensor}
        #   example; schlieren -> (1,80,256), target (1,80,256), label -> (3,)

# DataLoader creation with train/test split
def create_dataloaders(directory, cleaned_datafiles, target_variables, stats_dict, batch_size,seed = training_config.seed, shuffle=True):
    #reproduce same test sets
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    
    dataset = CFD_preprocessing(directory, cleaned_datafiles, target_variables, stats_dict)
    

    train_size = int(training_config.train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    # consistent generator acc to seed. 
    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    return train_dataset, test_dataset

