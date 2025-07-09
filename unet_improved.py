# This U-Net is for the improved DDPM variation which outputs the interpolation weight v as well as noise prediction

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import training_config
from model.layers_cond_binary import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionalEmbedding, ConvUpBlock, ResNetBlock
#from model.layers_cond_binary_improved import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionalEmbedding, ConvUpBlock, ResNetBlock
import torch.nn.init as init  # Weight initialization


class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B.
    """

    def __init__(self, num_conditions=3, input_channels=training_config.input_channels, condition_channels=training_config.condition_channels):
        super().__init__()

        #  Gradient embedding (for `cond_map`)
        self.emb_conv = nn.Sequential(
            nn.Conv2d(condition_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        )
        
        #  Initial convolution
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        init.xavier_uniform_(self.initial_conv.weight)

        #  Only initialize Conv layers (not MLP layers)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)

        self.combine_conv = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1, padding=0)

        #  Encode `cond_info` (FiLM-style conditioning)
        self.cond_encoding = nn.Sequential(
            nn.Linear(num_conditions, 128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        #  Encode diffusion timestep (`time`)
        self.time_positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

        #  Downsampling blocks
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvDownBlock(in_channels=256, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4)
        ])

        #  Bottleneck (Self-Attention Block)
        self.bottleneck = AttentionDownBlock(
            in_channels=512, out_channels=512, num_layers=2, num_att_heads=4, num_groups=32,
            time_emb_channels=128 * 4, cond_emb_channels=128 * 4, downsample=False
        )

        #  Upsampling blocks
        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=512 + 512, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            AttentionUpBlock(in_channels=512 + 256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4)
        ])

        #  Output convolution (Predicts both noise & variance)
        self.output_conv = nn.Sequential(
            
            nn.GroupNorm(num_channels=128, num_groups=32),  # ✅ Fix: num_channels=128
            nn.SiLU(),
            nn.Conv2d(128, 2, 3, padding=1)  # ✅ 2 output channels for noise & variance
        )
        nn.init.zeros_(self.output_conv[2].bias)  # Initialize final conv bias to zero


        #self.output_features = nn.Sequential(
        #    nn.GroupNorm(num_channels=256, num_groups=32),
        #    nn.SiLU()   
        #)
        #self.mean_output = nn.Conv2d(256, 1, 3, padding=1)
        #self.variance_output = nn.Conv2d(256, 1, 3, padding=1)

   # ✅ Fix: num_channels=128
    def forward(self, input_tensor, time, cond_info, cond_map=None):
        """
        Forward pass through the U-Net.
        """
        print(f" Stats of U-Net INPUT: min={input_tensor.min().item()}, max={input_tensor.max().item()}, mean={input_tensor.mean().item()}, variance={input_tensor.var().item()}")

        x = self.initial_conv(input_tensor)

        # Inject `cond_map` spatial conditioning
        if cond_map is not None:
            cond_emb = self.emb_conv(cond_map)
        else:
            cond_emb = torch.zeros_like(x)

        x = torch.cat((x, cond_emb), dim=1)
        x = self.combine_conv(x)

        #  Process `cond_info` & `time` embeddings
        cond_info_encoded = self.cond_encoding(cond_info)
        time_encoded = self.time_positional_encoding(time)

        #  Skip connections
        states_for_skip_connections = [x]

        for block in self.downsample_blocks:
            x = block(x, time_encoded, cond_info_encoded)
            states_for_skip_connections.append(x)

        states_for_skip_connections = list(reversed(states_for_skip_connections))
        x = self.bottleneck(x, time_encoded, cond_info_encoded)

        for block, skip in zip(self.upsample_blocks, states_for_skip_connections):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded, cond_info_encoded)

        #  Output convolution
        out = self.output_conv(x)


#features = self.output_features(x)
#mean = self.mean_output(features)
#log_variance = self.variance_output(features)
#return mean, log_variance

        #predicted_noise, predicted_log_variance = out[:, 0:1, :, :], out[:, 1:2, :, :]

        predicted_noise = out[:, 0:1, :, :]

        v = torch.sigmoid(out[:, 1:2, :, :])

        v = v.mean(dim=[2, 3], keepdim=True)


        #  Debugging Prints
        print(f"U-Net Predicted Noise Stats: min={predicted_noise.min().item()}, max={predicted_noise.max().item()}, mean={predicted_noise.mean().item()}, variance={predicted_noise.var().item()}")
        print(f"U-Net Interpolation Weight v: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
        return predicted_noise, v
