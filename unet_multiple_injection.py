import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from train import training_config
from model.layers_cond_binary import ConvDownBlock, AttentionDownBlock, AttentionUpBlock, TransformerPositionalEmbedding, ConvUpBlock
import torch.nn.init as init

class UNet(nn.Module):
    """
    Model architecture as described in the DDPM paper, Appendix, section B
    Enhanced with multi-level conditioning
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
        
        # NEW: Condition map processors for multiple levels
        self.cond_processors = nn.ModuleList([
            nn.Conv2d(128, 128, 1),  # For first level
            nn.Conv2d(128, 256, 1),  # For second level
            nn.Conv2d(128, 256, 1),  # For third level
            nn.Conv2d(128, 512, 1)   # For fourth level
        ])
        
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
        
        # Process condition map and create multi-resolution versions
        if cond_map is not None:
            base_cond_feature = self.emb_conv(cond_map)
            # Store original for initial concatenation
            original_cond_emb = base_cond_feature
            
            # Create a list of downsampled features for each resolution
            cond_features = [base_cond_feature]
            for i in range(len(self.downsample_blocks)-1):
                # Create progressively downsampled versions - downsample the previous level
                # to ensure correct size progression
                downsampled = F.avg_pool2d(cond_features[-1], 2)
                cond_features.append(downsampled)
        else:
            cond_features = [None] * len(self.downsample_blocks)
            original_cond_emb = torch.zeros_like(x)
        
        # Initial concatenation (as in original code)
        x = torch.cat((x, original_cond_emb), dim=1)
        x = self.combine_conv(x)

        # Generate embeddings
        cond_info_encoded = self.cond_encoding(cond_info)    
        time_encoded = self.time_positional_encoding(time)      
 
        states_for_skip_connections = [x]

        # Downsampling path with multi-level conditioning
        for i, block in enumerate(self.downsample_blocks):
            # Apply regular downsampling block
            x = block(x, time_encoded, cond_info_encoded)
            
            # Add spatial condition at this level if available
            if cond_map is not None:
                # Process the condition feature to match current channels
                processed_cond = self.cond_processors[i](cond_features[i])
                # Ensure spatial dimensions match before adding
                if processed_cond.shape[2:] != x.shape[2:]:
                    processed_cond = F.interpolate(processed_cond, size=x.shape[2:], mode='bilinear', align_corners=True)
                # Simple addition of processed condition
                x = x + processed_cond
                
            states_for_skip_connections.append(x)
            
        states_for_skip_connections = list(reversed(states_for_skip_connections))
        x = self.bottleneck(x, time_encoded, cond_info_encoded)

        for i, (block, skip) in enumerate(zip(self.upsample_blocks, states_for_skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = block(x, time_encoded, cond_info_encoded)
           
        # Concat initial_conv with tensor
        x = torch.cat([x, states_for_skip_connections[-1]], dim=1)
        
        out = self.output_conv(x)

        # **DEBUGGING OUTPUTS**
        print(f"U-Net Predicted Noise Stats : min={out.min().item()}, max={out.max().item()}, mean={out.mean().item()}, variance={out.var().item()}")

        return out
    