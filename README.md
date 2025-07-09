High-Fidelity Supersonic Flow Reconstruction with Conditional Diffusion Models
This repository contains the codebase developed for my master’s thesis at the Technical University of Munich, titled "High-Fidelity Supersonic Flow Field Reconstruction using Conditional Denoising Diffusion Probabilistic Models"

The project implements a Conditional Denoising Diffusion Probabilistic Model (DDPM) to reconstruct high-resolution supersonic flow fields (velocity, Mach number, density) from Schlieren images and experimental boundary conditions (pressure, temperature, gas type). The model leverages:
->A U-Net architecture with FiLM conditioning and multi-resolution Schlieren injection
->Cosine-based noise scheduling
->A custom hybrid loss combining Mean Squared Error (MSE) and Structural Similarity Index (SSIM)
->Optional enhancements including non-uniform timestep sampling and attention layers
