# Standard Unet implementation for conditional diffusion model

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from train import training_config
from model.layers_cond_binary import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionalEmbedding, ConvUpBlock
import torch.nn.init as init #trying some initialisation stuff

class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    """

    def __init__(self, num_conditions=3, input_channels=training_config.input_channels, condition_channels=training_config.condition_channels):
        super().__init__()
        # 1. We replaced weight normalization with group normalization
        # 2. Our 32x32 models use four feature map resolutions (32x32 to 4x4), and our 256x256 models use six (I made 5)
        # 3. Two convolutional residual blocks per resolution level and self-attention blocks at the 16x16 resolution
        # between the convolutional blocks [https://arxiv.org/pdf/1712.09763.pdf]
        # 4. Diffusion time t is specified by adding the Transformer sinusoidal position embedding into
        # each residual block [https://arxiv.org/pdf/1706.03762.pdf]
        
        # gradient embedding
        self.emb_conv = nn.Sequential(
            torch.nn.Conv2d(condition_channels, 128, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        )
        
        self.initial_conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='circular')
        init.xavier_uniform_(self.initial_conv.weight)  # 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):  
                init.xavier_uniform_(m.weight)

        self.combine_conv = torch.nn.Conv2d(128*2, 128, kernel_size=1, stride=1, padding=0)

        self.cond_encoding = nn.Sequential(
            nn.Linear(num_conditions, 128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )
              
        self.time_positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(dimension=128),
            nn.Linear(128, 128 * 4),
            nn.GELU(),
            nn.Linear(128 * 4, 128 * 4)
        )

             
        self.downsample_blocks = nn.ModuleList([
            ConvDownBlock(in_channels=128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvDownBlock(in_channels=128, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            AttentionDownBlock(in_channels=256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvDownBlock(in_channels=256, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4)
        ])
     
        self.bottleneck = AttentionDownBlock(in_channels=512, out_channels=512, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128*4, cond_emb_channels=128 * 4, downsample=False)                                                                                                  # 16x16x256 -> 16x16x256

        self.upsample_blocks = nn.ModuleList([
            ConvUpBlock(in_channels=512 + 512, out_channels=512, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            AttentionUpBlock(in_channels=512 + 256, out_channels=256, num_layers=2, num_att_heads=4, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 256, out_channels=256, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4),
            ConvUpBlock(in_channels=256 + 128, out_channels=128, num_layers=2, num_groups=32, time_emb_channels=128 * 4, cond_emb_channels=128 * 4)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_channels=256, num_groups=32),
            nn.SiLU(),
            nn.Conv2d(256, 1, 3, padding=1)
            
        )

    def forward(self, input_tensor, time, cond_info, cond_map=None):
        

        print(f" Stats of U Net INPUT: min={input_tensor.min().item()}, max={input_tensor.max().item()}, mean={input_tensor.mean().item()}, variance={input_tensor.var().item()}")

        x = self.initial_conv(input_tensor)
        #print(f"Input shape: {x.shape}")  # Log the input shape

        
        if cond_map is not None:
            #print(f"SHape ph condition : {cond_map.shape}")
            cond_emb = self.emb_conv(cond_map)
          #  print(f"Conditional Embedding Shape: {cond_emb.shape}")
        else:
            cond_emb = torch.zeros_like(x)
            
        
        x = torch.cat((x, cond_emb), dim=1)
        #print(f"After combining input and condition: {x.shape}")


        x = self.combine_conv(x)
        #print(f"After combine_conv: {x.shape}")


        cond_info_encoded = self.cond_encoding(cond_info)    
        time_encoded = self.time_positional_encoding(time)      
        #print(f"Encoded Condition Shape: {cond_info_encoded.shape}, Encoded Time Shape: {time_encoded.shape}")
 
        states_for_skip_connections = [x]

        for i, block in enumerate(self.downsample_blocks):
            x = block(x, time_encoded, cond_info_encoded)
            #print(f"After downsample block {i}: {x.shape}")
            states_for_skip_connections.append(x)
        states_for_skip_connections = list(reversed(states_for_skip_connections))
        x = self.bottleneck(x, time_encoded, cond_info_encoded)
        #print(f"After bottleneck: {x.shape}")


        for i, (block, skip) in enumerate(zip(self.upsample_blocks, states_for_skip_connections)):
            #print(f"Upsampling block {i} - Input shape: {x.shape}, Skip connection shape: {skip.shape}")

            #print(f"Before concat at block {i}: x shape: {x.shape}, skip shape: {skip.shape}") (debugging, fixed)
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded, cond_info_encoded)
            #print(f"After upsample block {i}: {x.shape}")
           
        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        #print(f"After final concatenation: {x.shape}")
        
        
        out = self.output_conv(x)

        # **DEBUGGING OUTPUTS**
        print(f"U-Net Predicted Noise Stats : min={out.min().item()}, max={out.max().item()}, mean={out.mean().item()}, variance={out.var().item()}")
        # noise is mean 0 and unit variance so lets see in debugging whats going on..

        return out
