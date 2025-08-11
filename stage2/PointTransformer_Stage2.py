import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 3D_pointtransformer 경로 추가 (lib.pointops 모듈을 위해)
sys.path.append('/home/jhrew/jiye/3D_pointtransformer')

# Stage 1의 PointTransformer 모델 import - 상대 경로 사용
sys.path.append('..')  # stage2 디렉토리에서 상위 디렉토리로
from PointTransformer_model import PointTransformerLandmark

##############################################
class GlobalFeatureExtractor(nn.Module):
    """Improved Global Feature Extractor using Avg+Max+Std+GeM pooling"""
    def __init__(self, landmark_num=68, feature_dim=512):
        super().__init__()
        self.landmark_num = landmark_num
        self.feature_dim = feature_dim
        
        # GeM pooling parameter (learnable)
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        
        # Feature projection MLP
        # Input: Avg(68) + Max(68) + Std(68) + GeM(68) = 272
        input_dim = landmark_num * 4  # 68 * 4 = 272
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (B, 68, N) - heatmap predictions from Stage 1
        Returns:
            global_features: (B, 512) - extracted global features
        """
        batch_size = heatmaps.size(0)
        
        # 1. Average Pooling across points
        avg_features = torch.mean(heatmaps, dim=2)  # (B, 68)
        
        # 2. Max Pooling across points
        max_features = torch.max(heatmaps, dim=2)[0]  # (B, 68)
        
        # 3. Standard Deviation across points
        std_features = torch.std(heatmaps, dim=2)  # (B, 68)
        
        # 4. GeM (Generalized Mean) Pooling across points
        # GeM: (1/N * sum(x^p))^(1/p) where p is learnable
        p = self.gem_p.clamp(min=1e-6, max=10.0)  # Clamp p to avoid numerical issues
        gem_features = torch.pow(torch.mean(torch.pow(heatmaps, p), dim=2), 1.0/p)  # (B, 68)
        
        # 5. Concatenate all pooling features
        pooled_features = torch.cat([
            avg_features,    # (B, 68)
            max_features,    # (B, 68)
            std_features,    # (B, 68)
            gem_features     # (B, 68)
        ], dim=1)  # (B, 272)
        
        # 6. Project to final feature dimension
        global_features = self.feature_proj(pooled_features)  # (B, 272) → (B, 512)
        
        return global_features

##############################################
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
        
        # Improved Global Feature Extractor
        self.global_feature_extractor = GlobalFeatureExtractor(
            landmark_num=landmark_num,
            feature_dim=512
        )
        
        # Coarse-to-Fine Regression Head
        self.coarse_to_fine_head = CoarseToFineHead(
            input_dim=512,  # Global feature dimension
            landmark_num=landmark_num
        )
        
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
        
        # Extract improved global features using Avg+Max+Std+GeM + MLP
        global_features = self.global_feature_extractor(heatmaps)  # (B, 512)
        
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