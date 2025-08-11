#!/usr/bin/env python3
"""
OBJ 메쉬 파일에서 3D 랜드마크 예측
사용법: python predict_landmarks.py input.obj output_landmarks.npy
"""

import torch
import numpy as np
import os
import sys
import argparse

# 필수 라이브러리 체크
try:
    import trimesh
except ImportError:
    print("❌ trimesh 설치 필요: pip install trimesh")
    sys.exit(1)

# Stage 2 모델 import
sys.path.append('/home/jhrew/jiye/3D_landmark')
from My_args import *
from PointTransformer_Stage2 import PointTransformerStage2

def obj_to_pointcloud(obj_path, num_points=8192):
    """OBJ 파일을 포인트 클라우드로 변환"""
    mesh = trimesh.load(obj_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = list(mesh.geometry.values())[0]
    
    points = mesh.sample(num_points)
    print(f"✅ 포인트 클라우드 생성: {points.shape}")
    return points

def normalize_points(points):
    """포인트 클라우드 정규화"""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / max_dist
    return points_normalized, centroid, max_dist

def load_model(model_path):
    """모델 로드"""
    class Args:
        def __init__(self):
            self.use_unfolding_loss = False
            self.use_registration_loss = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointTransformerStage2(Args(), 68, pretrained_path=None)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # DataParallel 처리
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        new_state_dict[key] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model.to(device), device

def predict(model, points_normalized, device):
    """랜드마크 예측"""
    points_tensor = torch.FloatTensor(points_normalized).unsqueeze(0).to(device)
    points_tensor = points_tensor.permute(0, 2, 1)  # (1, 3, N)
    
    with torch.no_grad():
        _, _, refined_landmarks = model(points_tensor)
        return refined_landmarks.cpu().numpy().squeeze(0)  # (68, 3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_file', help='입력 OBJ 파일')
    parser.add_argument('output_file', help='출력 NPY 파일')
    parser.add_argument('--model', default='./checkpoints/best_model_stage2.t7', help='모델 파일')
    args = parser.parse_args()
    
    print(f"🚀 {args.obj_file} → {args.output_file}")
    
    # 1. OBJ → 포인트 클라우드
    points = obj_to_pointcloud(args.obj_file)
    
    # 2. 정규화
    points_norm, centroid, max_dist = normalize_points(points)
    
    # 3. 모델 로드 및 예측
    model, device = load_model(args.model)
    landmarks_norm = predict(model, points_norm, device)
    
    # 4. 역정규화
    landmarks = landmarks_norm * max_dist + centroid
    
    # 5. 저장
    result = {
        'landmarks': landmarks,  # (68, 3) - 최종 랜드마크
        'points': points,        # (8192, 3) - 원본 포인트 클라우드
        'obj_file': args.obj_file
    }
    np.save(args.output_file, result)
    
    print(f"✅ 완료! 랜드마크 {landmarks.shape[0]}개 예측")
    print(f"💾 저장: {args.output_file}")

if __name__ == '__main__':
    main()