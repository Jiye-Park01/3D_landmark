import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 3D_pointtransformer 경로 추가 (lib.pointops 모듈을 위해)
sys.path.append('/home/jhrew/jiye/3D_pointtransformer')

# Stage 1의 PointTransformer 모델 import
sys.path.append('/home/jhrew/jiye/3D_landmark')
from PointTransformer_model import PointTransformerLandmark

class CoarseToFineHead(nn.Module):
    """Coarse-to-Fine Regression Head for precise 3D landmark prediction"""
    def __init__(self, input_dim, landmark_num, hidden_dim=256):
        super(CoarseToFineHead, self).__init__()
        self.landmark_num = landmark_num
        self.input_dim = input_dim
        
        # Coarse MLP: 초기 좌표 예측
        self.coarse_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, landmark_num * 3)  # (B, L*3)
        )
        
        # Refinement MLP: 정밀 좌표 보정
        # 입력: features (input_dim) + coarse coordinates (landmark_num * 3)
        refinement_input_dim = input_dim + landmark_num * 3
        self.refinement_mlp = nn.Sequential(
            nn.Linear(refinement_input_dim, hidden_dim),  # 실제 입력 차원에 맞춤
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, landmark_num * 3)  # (B, L*3)
        )
        
    def forward(self, features, coarse_landmarks=None):
        """
        Args:
            features: (B, C) - global features from backbone
            coarse_landmarks: (B, L, 3) - coarse predictions (optional for first pass)
        """
        batch_size = features.size(0)
        
        # Coarse prediction
        coarse_out = self.coarse_mlp(features)  # (B, L*3)
        coarse_landmarks = coarse_out.view(batch_size, self.landmark_num, 3)  # (B, L, 3)
        
        # Refinement: coarse coordinates + features
        if coarse_landmarks is None:
            # First pass: use zeros as initial coarse
            coarse_input = torch.zeros_like(coarse_landmarks)
        else:
            coarse_input = coarse_landmarks
            
        # Concatenate features with coarse coordinates
        coarse_flat = coarse_input.view(batch_size, -1)  # (B, L*3)
        refined_input = torch.cat([features, coarse_flat], dim=1)  # (B, C + L*3)
        
        # Refinement prediction
        refinement_out = self.refinement_mlp(refined_input)  # (B, L*3)
        refined_landmarks = refinement_out.view(batch_size, self.landmark_num, 3)  # (B, L, 3)
        
        return coarse_landmarks, refined_landmarks

class PointTransformerStage2(nn.Module):
    """Stage 2: PointTransformer with Heatmap + Coarse-to-Fine Regression"""
    
    def __init__(self, args, landmark_num, pretrained_path=None):
        super(PointTransformerStage2, self).__init__()
        
        # Load Stage 1 model as backbone
        self.stage1_model = PointTransformerLandmark(args, landmark_num, pretrained_path)
        
        # Freeze backbone initially (optional, can be unfrozen later)
        if hasattr(self.stage1_model, 'freeze_backbone'):
            self.stage1_model.freeze_backbone()
        
        # Get feature dimension from Stage 1 model
        # Assuming Stage 1 model has a global feature extractor
        feature_dim = 512  # Adjust based on your Stage 1 model's feature dimension
        
        # Coarse-to-Fine Regression Head
        self.coarse_to_fine_head = CoarseToFineHead(
            input_dim=feature_dim,
            landmark_num=landmark_num
        )
        
        # Global feature extractor (if needed)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 3, N) - normalized point cloud
        Returns:
            heatmaps: (B, L, N) - heatmap predictions
            coarse_landmarks: (B, L, 3) - coarse landmark coordinates
            refined_landmarks: (B, L, 3) - refined landmark coordinates
        """
        # Get heatmap predictions from Stage 1 model
        heatmaps = self.stage1_model(x)  # (B, L, N)
        
        # Extract global features for regression
        # Use the last layer features from Stage 1 model
        # This might need adjustment based on your Stage 1 model architecture
        batch_size = x.size(0)
        
        # For now, we'll use a simple approach: average pooling of heatmaps
        # In practice, you might want to extract features from intermediate layers
        global_features = torch.mean(heatmaps, dim=2)  # (B, L)
        global_features = torch.mean(global_features, dim=1, keepdim=True)  # (B, 1)
        
        # Expand to match expected feature dimension
        # This is a placeholder - you should extract actual features from Stage 1
        global_features = global_features.expand(-1, 512)  # (B, 512)
        
        # Coarse-to-Fine regression
        coarse_landmarks, refined_landmarks = self.coarse_to_fine_head(global_features)
        
        return heatmaps, coarse_landmarks, refined_landmarks
    
    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning"""
        if hasattr(self.stage1_model, 'unfreeze_backbone'):
            self.stage1_model.unfreeze_backbone()
        else:
            # If no specific unfreeze method, unfreeze all parameters
            for param in self.stage1_model.parameters():
                param.requires_grad = True
    
    def freeze_backbone(self):
        """Freeze the backbone"""
        if hasattr(self.stage1_model, 'freeze_backbone'):
            self.stage1_model.freeze_backbone()
        else:
            # If no specific freeze method, freeze all parameters
            for param in self.stage1_model.parameters():
                param.requires_grad = False 