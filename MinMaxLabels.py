"""
This code is used to find the Min and the Max value of the pressure and temprature from the file names.
This is used for normalizing the labels.
"""

import os
import re

# Path to the folder containing .npz files
folder_path = "/local/disk/home/kbharadwaj/Thesis/Thesis/npz_files" 
# Initialize variables to store min and max
min_pressure = float('inf')
max_pressure = float('-inf')
min_temperature = float('inf')
max_temperature = float('-inf')

# Regex patterns to extract pressure and temperature
pressure_pattern = re.compile(r"press_([\d.]+)bar")
temperature_pattern = re.compile(r"temp_([\d.]+)K")

# Iterate through all .npz files in the folder
npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]

for file_name in npz_files:
    # Extract pressure
    pressure_match = pressure_pattern.search(file_name)
    if pressure_match:
        pressure = float(pressure_match.group(1))
        min_pressure = min(min_pressure, pressure)
        max_pressure = max(max_pressure, pressure)
    
    # Extract temperature
    temperature_match = temperature_pattern.search(file_name)
    if temperature_match:
        temperature = float(temperature_match.group(1))
        min_temperature = min(min_temperature, temperature)
        max_temperature = max(max_temperature, temperature)

# Print results
print(f"Pressure: Min = {min_pressure:.2f} bar, Max = {max_pressure:.2f} bar")
print(f"Temperature: Min = {min_temperature:.2f} K, Max = {max_temperature:.2f} K")
