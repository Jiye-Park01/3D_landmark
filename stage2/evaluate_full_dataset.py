#!/usr/bin/env python3
"""
전체 테스트 데이터셋에 대한 Stage 2 모델 평가 스크립트
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import argparse
from datetime import datetime

# Stage 1 모델 import
sys.path.append('/home/jhrew/jiye/3D_landmark')
from My_args import *
from PointTransformer_Stage2 import PointTransformerStage2
from dataset_stage2 import FaceLandmarkDataStage2, custom_collate_fn_stage2

def normalize_data(points, landmarks=None):
    """포인트 클라우드와 랜드마크 정규화 (중심 이동만, 스케일링 제거)"""
    centroid = torch.mean(points, dim=1, keepdim=True)
    points = points - centroid
    
    if landmarks is not None:
        landmarks = landmarks - centroid
        return points, landmarks
    
    return points

def calculate_landmark_error(pred_landmarks, true_landmarks):
    """랜드마크 오차 계산"""
    distances = torch.norm(pred_landmarks - true_landmarks, dim=2)
    mean_error = torch.mean(distances)
    return mean_error.item()

def calculate_nme(pred_landmarks, true_landmarks, face_size):
    """Normalized Mean Error 계산"""
    distances = torch.norm(pred_landmarks - true_landmarks, dim=2)
    nme = torch.mean(distances / face_size)
    return nme.item()

def evaluate_full_dataset(model, test_loader, device):
    """전체 테스트 데이터셋 평가"""
    model.eval()
    total_heatmap_loss = 0
    total_regression_loss = 0
    total_landmark_error = 0
    total_nme = 0
    num_batches = 0
    num_samples = 0
    
    # 개별 샘플 성능 추적용 리스트
    sample_results = []
    
    heatmap_criterion = nn.MSELoss()
    regression_criterion = nn.MSELoss()
    
    print("Starting full dataset evaluation...")
    
    with torch.no_grad():
        for batch_idx, (points, landmarks, true_heatmaps) in enumerate(test_loader):
            points, landmarks, true_heatmaps = points.to(device), landmarks.to(device), true_heatmaps.to(device)
            batch_size = points.size(0)
            
            # 정규화 (포인트와 랜드마크 모두)
            points_normal, landmarks_normal = normalize_data(points, landmarks)
            points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 모델 예측
            pred_heatmaps, coarse_landmarks, refined_landmarks = model(points_normal)
            
            # 1. Heatmap Loss
            heatmap_loss = heatmap_criterion(pred_heatmaps, true_heatmaps)
            
            # 2. Regression Loss (refined landmarks 사용, 정규화된 좌표계)
            regression_loss = regression_criterion(refined_landmarks, landmarks_normal)
            
            # 3. Landmark Error 계산 (정규화된 좌표계)
            landmark_error = calculate_landmark_error(refined_landmarks, landmarks_normal)
            
            # 4. NME 계산 (정규화된 좌표계에서)
            face_size = torch.max(torch.norm(landmarks_normal, dim=2))  # 얼굴 크기 추정
            nme = calculate_nme(refined_landmarks, landmarks_normal, face_size)
            
            # 배치 내 각 샘플의 개별 성능 저장
            for i in range(batch_size):
                sample_landmark_error = torch.norm(refined_landmarks[i] - landmarks_normal[i], dim=1).mean().item()
                sample_nme = torch.norm(refined_landmarks[i] - landmarks_normal[i], dim=1).mean().item() / torch.max(torch.norm(landmarks_normal[i], dim=1)).item()
                
                # 샘플 인덱스 계산 (전체 데이터셋에서의 위치)
                global_sample_idx = batch_idx * test_loader.batch_size + i
                
                sample_results.append({
                    'sample_idx': global_sample_idx,
                    'batch_idx': batch_idx,
                    'batch_position': i,
                    'landmark_error': sample_landmark_error,
                    'nme': sample_nme
                })
            
            total_heatmap_loss += heatmap_loss.item()
            total_regression_loss += regression_loss.item()
            total_landmark_error += landmark_error
            total_nme += nme
            num_batches += 1
            num_samples += batch_size
            
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)} ({num_samples} samples)")
    
    # 평균 계산
    avg_heatmap_loss = total_heatmap_loss / num_batches
    avg_regression_loss = total_regression_loss / num_batches
    avg_landmark_error = total_landmark_error / num_batches
    avg_nme = total_nme / num_batches
    
    # 성능 기준으로 상위 5개 샘플 찾기 (NME 기준으로 정렬)
    sample_results.sort(key=lambda x: x['nme'])
    top5_best = sample_results[:5]
    
    return avg_heatmap_loss, avg_regression_loss, avg_landmark_error, avg_nme, num_samples, top5_best

def load_model(model_path, args, landmark_num, device):
    """모델 로드"""
    model = PointTransformerStage2(args, landmark_num, pretrained_path=None)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # state_dict 추출
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # DataParallel로 저장된 경우 'module.' 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/best_model_stage2.t7',
                       help='Stage 2 모델 경로')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='테스트 데이터 디렉토리')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='배치 크기')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='데이터 로더 워커 수')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    test_dataset = FaceLandmarkDataStage2(
        data_dir=args.data_dir,
        sigma=1.0,
        num_points=8192,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn_stage2,
        drop_last=False
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    print(f"Test loader: {len(test_loader)} batches")
    
    # 모델 로드
    landmark_num = test_dataset.get_landmark_num()
    model = load_model(args.model_path, args, landmark_num, device)
    model = model.to(device)
    
    print(f"Loaded Stage 2 model with {landmark_num} landmarks")
    
    # 전체 데이터셋 평가
    start_time = datetime.now()
    avg_heatmap_loss, avg_regression_loss, avg_landmark_error, avg_nme, total_samples, top5_best = evaluate_full_dataset(
        model, test_loader, device
    )
    end_time = datetime.now()
    
    # 결과 출력
    print(f"\n=== Full Dataset Evaluation Results ===")
    print(f"Evaluation time: {end_time - start_time}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Average Heatmap Loss: {avg_heatmap_loss:.6f}")
    print(f"Average Regression Loss: {avg_regression_loss:.6f}")
    print(f"Average Landmark Error: {avg_landmark_error:.6f}")
    print(f"Average NME: {avg_nme:.6f}")
    
    # 결과를 파일로 저장
    results_file = os.path.join(args.output_dir, 'full_dataset_evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Stage 2 Full Dataset Evaluation Results\n")
        f.write(f"=====================================\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Data: {args.data_dir}\n")
        f.write(f"Evaluation time: {end_time - start_time}\n")
        f.write(f"Total samples evaluated: {total_samples}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"\nMetrics:\n")
        f.write(f"Average Heatmap Loss: {avg_heatmap_loss:.8f}\n")
        f.write(f"Average Regression Loss: {avg_regression_loss:.8f}\n")
        f.write(f"Average Landmark Error: {avg_landmark_error:.8f}\n")
        f.write(f"Average NME: {avg_nme:.8f}\n")
        
        # Top 5 최고 성능 샘플 정보 추가
        f.write(f"\n" + "="*50 + "\n")
        f.write(f"TOP 5 BEST PERFORMING SAMPLES (lowest NME)\n")
        f.write(f"="*50 + "\n")
        for i, sample in enumerate(top5_best, 1):
            f.write(f"\nRank {i}:\n")
            f.write(f"  Sample Index: {sample['sample_idx']}\n")
            f.write(f"  Batch Index: {sample['batch_idx']}\n")
            f.write(f"  Batch Position: {sample['batch_position']}\n")
            f.write(f"  Landmark Error: {sample['landmark_error']:.6f}\n")
            f.write(f"  NME: {sample['nme']:.6f}\n")
            
            # 테스트 데이터셋에서 실제 파일 정보 가져오기 (가능한 경우)
            try:
                actual_idx = test_dataset.indices[sample['sample_idx']]
                pc_file, landmark_file = test_dataset.data_pairs[actual_idx]
                pc_basename = os.path.basename(pc_file)
                landmark_basename = os.path.basename(landmark_file)
                person_id = landmark_basename.split('_')[0]
                expression_id = landmark_basename.split('_')[2].split('.')[0]
                
                f.write(f"  Person ID: {person_id}\n")
                f.write(f"  Expression: {expression_id}\n")
                f.write(f"  PC File: {pc_basename}\n")
                f.write(f"  Landmark File: {landmark_basename}\n")
            except:
                f.write(f"  File info: Could not retrieve\n")
    
    print(f"\nResults saved to {results_file}")
    
    # CSV 형태로도 저장
    csv_file = os.path.join(args.output_dir, 'full_dataset_evaluation_results.csv')
    with open(csv_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Total_Samples,{total_samples}\n")
        f.write(f"Heatmap_Loss,{avg_heatmap_loss:.8f}\n")
        f.write(f"Regression_Loss,{avg_regression_loss:.8f}\n")
        f.write(f"Landmark_Error,{avg_landmark_error:.8f}\n")
        f.write(f"NME,{avg_nme:.8f}\n")
    
    print(f"CSV results saved to {csv_file}")
    
    # Top 5 결과 출력
    print(f"\n=== TOP 5 BEST PERFORMING SAMPLES ===")
    for i, sample in enumerate(top5_best, 1):
        print(f"Rank {i}: Sample {sample['sample_idx']}, NME: {sample['nme']:.6f}, Landmark Error: {sample['landmark_error']:.6f}")

if __name__ == '__main__':
    main()