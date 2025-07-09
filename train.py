from dataclasses import dataclass
import os

@dataclass
class TrainingConfig:

    ############################################################################################################
    # Directories
    experiment_name = "spatial_error"
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", experiment_name))
    dataset_path = "/local/disk/home/kbharadwaj/Thesis/Thesis/npz_files"
    cleaned_output_path = os.path.join(os.path.dirname(dataset_path), "cleaned_data")
    stats_folder = os.path.join(base_path, "stats")
    outliers_path = os.path.join(stats_folder, "outliers.csv")
    stats_output_path = os.path.join(stats_folder, "Stats.xlsx")
    processed_npz_path = "/local/disk/home/kbharadwaj/Thesis/Thesis/pipeline/data_preprocessing/processed_npz_path"
    ############################################################################################################
    
    model_name = 'ddpm'
    dataset = 'nozzle_data'

    ############################################################################################################
    # Data Preprocessing
    processing_data = "normalization_then_scaling"
    scaling_min = -1.0
    scaling_max = 1.0
    train_ratio = 0.9 # 
    crop_half = "False"
    ############################################################################################################

    ############################################################################################################
    # U Net 
    condition_channels = 1
    target_variables = ['velocity_magnitude']
    input_channels = len(target_variables)
    condition_variable = "schlieren"
    ############################################################################################################

    ############################################################################################################
    # DDPM 
    diffusion_timesteps = 1000
    image_size = [80, 256]
    ############################################################################################################

    ############################################################################################################
    # Training Parameters
    train_batch_size = 32
    eval_batch_size = 64
    num_epochs = 200 # fine tuning
    start_epoch = 0
    learning_rate = 2e-5
    save_image_epochs = 100  #50 is good normally
    save_model_epochs = 100
    ############################################################################################################    
    
    device = "cuda"
    seed = 42
    version = 1
    resume = None
    

training_config = TrainingConfig()