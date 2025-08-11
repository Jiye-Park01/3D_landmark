import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from My_args import *

# Point Transformer import를 위한 경로 추가
sys.path.append('/home/jhrew/jiye/3D_pointtransformer/model/pointtransformer')
from pointtransformer_seg import pointtransformer_seg_repro

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_landmarks, sigma=10):
        super(HeatmapHead, self).__init__()
        self.num_landmarks = num_landmarks
        self.sigma = sigma
        
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, num_landmarks, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class PointTransformerLandmark(nn.Module):
    def __init__(self, args, landmark_num, pretrained_path=None):
        super(PointTransformerLandmark, self).__init__()
        self.args = args
        self.landmark_num = landmark_num
        
        # Point Transformer backbone (encoder only)
        self.backbone = pointtransformer_seg_repro(
            c=3,  # 3D coordinates only
            k=args.k if hasattr(args, 'k') else 13,
            num_points=args.num_points if hasattr(args, 'num_points') else 2048,
            use_decoder=False  # Only use encoder for feature extraction
        )
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Filter out decoder weights and load only encoder weights
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # Only load encoder-related weights (exclude decoder parts)
                if not any(decoder_key in key for decoder_key in ['folding_decoder', 'global_pool', 'latent_proj']):
                    encoder_state_dict[key] = value
            
            # Load the filtered state dict
            missing_keys, unexpected_keys = self.backbone.load_state_dict(encoder_state_dict, strict=False)
            print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            
            # Freeze backbone initially (freeze → unfreeze 전략)
            self.freeze_backbone()
        
        # Heatmap head for landmark prediction
        # Point Transformer encoder output is [N, planes[0]] where planes[0] = 32
        self.heatmap_head = HeatmapHead(288, landmark_num, sigma=args.sigma if hasattr(args, 'sigma') else 10)
        
        # Global feature projection for heatmap generation
        self.global_proj = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        
    def freeze_backbone(self):
        """Freeze the backbone encoder"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone encoder frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone encoder"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone encoder unfrozen")
    
    def forward(self, x):
        # x: (B, 3, N) where N is variable
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Convert to Point Transformer input format
        # Point Transformer expects: (n, 3), (n, c), (b)
        # where n is total points across all batches, c is features, b is batch offsets
        
        # Reshape input for Point Transformer
        x_reshaped = x.permute(0, 2, 1).contiguous()  # (B, N, 3)
        x_flat = x_reshaped.view(-1, 3)  # (B*N, 3)
        
        # Create batch offsets
        batch_offsets = torch.cumsum(torch.tensor([num_points] * batch_size, device=x.device), dim=0)
        
        # Point Transformer forward pass (encoder only)
        encoder_features = self.backbone([x_flat, x_flat, batch_offsets])  # (B*N, 32)
        
        # Reshape back to batch format
        encoder_features = encoder_features.view(batch_size, num_points, -1)  # (B, N, 32)
        encoder_features = encoder_features.permute(0, 2, 1)  # (B, 32, N)
        
        # Global feature extraction
        global_features = torch.max(encoder_features, dim=2, keepdim=True)[0]  # (B, 32, 1)
        global_features = global_features.squeeze(-1)  # (B, 32)
        
        # Project global features
        global_proj = self.global_proj(global_features)  # (B, 256)
        
        # Expand global features to match point features
        global_proj_expanded = global_proj.unsqueeze(-1).expand(-1, -1, num_points)  # (B, 256, N)
        
        # Concatenate local and global features
        combined_features = torch.cat([encoder_features, global_proj_expanded], dim=1)  # (B, 288, N)
        
        # Heatmap prediction
        heatmap = self.heatmap_head(combined_features)  # (B, num_landmarks, N)
        
        return heatmap

# Backward compatibility - keep PAConv class for existing code
# class PAConv(nn.Module):
#     def __init__(self, args, landmark_num, pretrained_path=None):
#         super(PAConv, self).__init__()
#         # Use Point Transformer instead of PAConv
#         self.model = PointTransformerLandmark(args, landmark_num, pretrained_path)
    
#     def forward(self, x):
#         return self.model(x)
    
#     def freeze_backbone(self):
#         return self.model.freeze_backbone()
    
#     def unfreeze_backbone(self):
#         return self.model.unfreeze_backbone()


