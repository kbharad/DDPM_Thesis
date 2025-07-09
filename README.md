High-Fidelity Supersonic Flow Reconstruction with Conditional Diffusion Models
This repository contains the codebase developed for my master’s thesis at the Technical University of Munich, titled "High-Fidelity Supersonic Flow Field Reconstruction using Conditional Denoising Diffusion Probabilistic Models"

The project implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) to reconstruct high-resolution supersonic flow fields (velocity, Mach number, density) from Schlieren images and experimental boundary conditions (pressure, temperature, gas type). The model leverages:

1) A U-Net architecture with FiLM conditioning and multi-resolution Schlieren injection

2) Cosine-based noise scheduling along with other options

3) Multiple loss functions including A custom hybrid loss combining Mean Squared Error (MSE) and Structural Similarity Index (SSIM)

4) Optional enhancements including non-uniform timestep sampling and attention layers
