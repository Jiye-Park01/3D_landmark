'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: train.py
@Time: 2021/12/02 09:59 AM
'''

import os
import math
import time
import numpy as np
from scipy import io
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import torch
import torch.nn as nn
import argparse
import random
from datetime import datetime
import shutil

from init import _init_, weight_init  # _init_ 함수를 명시적으로 import
from My_args import *
from augmentations import *
from dataset import FaceLandmarkData, custom_collate_fn
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    # Find the maximum number of points in the batch
    max_points = max(item[0].size(0) for item in batch)
    
    # Pad all point clouds to the maximum size
    padded_points = []
    for points, landmarks, _ in batch:
        # Create a padded tensor filled with zeros
        padded = torch.zeros(max_points, points.size(1))
        # Copy the actual points
        padded[:points.size(0)] = points
        padded_points.append(padded)
    
    # Stack the padded tensors
    points = torch.stack(padded_points)
    landmarks = torch.stack([item[1] for item in batch])
    
    return points, landmarks, None

def generate_heatmap(points, landmarks, sigma=2.0):
    """
    Generates Gaussian heatmap for landmarks on the point cloud.
    Args:
        points (torch.Tensor): Point cloud data (B, N, 3)
        landmarks (torch.Tensor): Landmark data (B, L, 3)
        sigma (float): Standard deviation for Gaussian kernel.
    Returns:
        torch.Tensor: Heatmap (B, L, N)
    """
    B, N, _ = points.shape
    _, L, _ = landmarks.shape
    
    # Reshape points for broadcasting (B, 1, N, 3)
    points = points.unsqueeze(1)
    # Reshape landmarks for broadcasting (B, L, 1, 3)
    landmarks = landmarks.unsqueeze(2)
    
    # Calculate squared Euclidean distance (B, L, N)
    # dist^2 = (px - lx)^2 + (py - ly)^2 + (pz - lz)^2
    dist_sq = torch.sum((points - landmarks)**2, dim=-1)
    
    # Calculate heatmap using Gaussian kernel
    # heatmap = exp(-dist_sq / (2 * sigma^2))
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    
    return heatmap

def get_predicted_landmarks_from_heatmap(points, pred_heatmap):
    """
    Extracts 3D landmark positions from predicted heatmap.
    Finds the point with the maximum heatmap value for each landmark.
    Args:
        points (torch.Tensor): Point cloud data (B, N, 3)
        pred_heatmap (torch.Tensor): Predicted heatmap (B, L, N)
    Returns:
        torch.Tensor: Predicted landmark positions (B, L, 3)
    """
    B, L, N = pred_heatmap.shape
    pred_landmarks = torch.zeros(B, L, 3).to(points.device)
    for b in range(B):
        for l in range(L):
            # Find the index of the point with the maximum heatmap value for landmark l
            max_idx = torch.argmax(pred_heatmap[b, l])
            # Get the 3D coordinate of this point
            pred_landmarks[b, l] = points[b, max_idx, :]
    return pred_landmarks

def evaluate(model, test_loader, criterion, device, epoch, args):
    model.eval()
    total_loss = 0
    total_landmark_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            points, landmark, _ = data
            points, landmark = points.to(device), landmark.to(device)
            
            # 모델 예측 (히트맵)
            # points는 (B, N, 3) 형태이므로, (B, 3, N)으로 변환
            points = points.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
            
            # PAConv 모델은 입력을 (B, 3, N) 형태로 기대함
            pred_heatmap = model(points)
            
            # 정답 히트맵 생성 (points를 다시 (B, N, 3)으로 변환)
            true_heatmap = generate_heatmap(points.permute(0, 2, 1), landmark, sigma=2.0)
            
            # 히트맵 손실 계산
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            total_loss += heatmap_loss.item()

            # 예측 랜드마크 위치 추출 (points를 다시 (B, N, 3)으로 변환)
            pred_landmarks = get_predicted_landmarks_from_heatmap(points.permute(0, 2, 1), pred_heatmap)

            # 랜드마크 오차 계산 (평균 거리)
            landmark_error = torch.norm(pred_landmarks - landmark, dim=2).mean()
            total_landmark_error += landmark_error.item()
            num_batches += 1

            if i == 0 and args.visualize and epoch % args.vis_interval == 0:
                # 시각화 디렉토리 생성
                vis_dir = os.path.join('./visualizations', args.exp_name, args.dataset)
                os.makedirs(vis_dir, exist_ok=True)
                
                # 첫 번째 샘플만 선택
                points_np = points[0].permute(1, 0).cpu().numpy()
                true_landmarks_np = landmark[0].cpu().numpy()
                pred_landmarks_np = pred_landmarks[0].cpu().numpy()
                
                # 3D 시각화
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 포인트 클라우드 (회색)
                ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], 
                        c='gray', s=1, alpha=0.1)
                
                # 실제 랜드마크 (빨간색)
                ax.scatter(true_landmarks_np[:, 0], true_landmarks_np[:, 1], true_landmarks_np[:, 2],
                        c='red', s=50, label='True Landmarks')
                
                # 예측 랜드마크 (파란색)
                ax.scatter(pred_landmarks_np[:, 0], pred_landmarks_np[:, 1], pred_landmarks_np[:, 2],
                        c='blue', s=50, label='Predicted Landmarks')
                
                # z축을 위에서 아래로 보이도록 설정
                ax.view_init(elev=90, azim=0)
                
                # 이미지 저장
                plt.savefig(os.path.join(vis_dir, f'prediction_epoch_{epoch+1}.png'))
                plt.close()
    
    avg_loss = total_loss / num_batches
    avg_landmark_error = total_landmark_error / num_batches
    
    return avg_loss, avg_landmark_error

def train(args):
    _init_(args)
    patience = 10  # 10 에포크 동안 개선이 없으면 중단
    min_delta = 0.0001  # 최소 개선 기준
    counter = 0  # 개선이 없을 때마다 증가
    best_metric = float('inf')  # 최고 성능 기록
    
    print(f"DEBUG: args.data_dir after _init_: {args.data_dir}")
    args.data_dir = './dataset'

    print(f"Number of points to sample: {args.num_points}")
    print(f"Number of landmarks: {args.num_landmarks}")

    train_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='train')
    test_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='val')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Dataset is empty after splitting. Check data_dir and file patterns.")
        return
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using device: {device}")

    model = PAConv(args, args.num_landmarks).to(device)
    model.apply(weight_init)

    if args.cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()  # Simple MSE loss for heatmap regression

    best_test_loss = float('inf')
    best_landmark_error = float('inf')

    log_dir = os.path.join('./checkpoints', args.exp_name, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'training_log.txt')
    
    if not os.path.exists(log_filepath) or os.stat(log_filepath).st_size == 0:
        with open(log_filepath, 'w') as f:
            f.write('Epoch\tTest Loss\tLandmark Error\n')

    print("Start training...")
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        model.train()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader):
            points, landmark, _ = data
            points, landmark = points.to(device), landmark.to(device)
            
            optimizer.zero_grad()

            # 모델 예측 (히트맵)
            # points는 (B, N, 3) 형태이므로, (B, 3, N)으로 변환
            points = points.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
            
            # PAConv 모델은 입력을 (B, 3, N) 형태로 기대함
            pred_heatmap = model(points)
            
            # 정답 히트맵 생성 (points를 다시 (B, N, 3)으로 변환)
            true_heatmap = generate_heatmap(points.permute(0, 2, 1), landmark, sigma=2.0)
            
            # Heatmap 손실 계산 (MSE Loss)
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            
            # 예측 랜드마크 위치 추출 (points를 다시 (B, N, 3)으로 변환)
            pred_landmarks = get_predicted_landmarks_from_heatmap(points.permute(0, 2, 1), pred_heatmap)

            # 랜드마크 위치 오차 계산 (L2 Loss)
            position_loss = torch.norm(pred_landmarks - landmark, dim=2).mean()

            # 총 손실 계산
            alpha = 0.1  # 논문에서 사용한 가중치
            total_loss = heatmap_loss + alpha * position_loss

            total_train_loss += total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            if (i + 1) % 10 == 0:
                print(f'  Iter {i+1}/{len(train_loader)}, Loss: {total_loss.item():.6f}')

        avg_test_loss, avg_landmark_error = evaluate(model, test_loader, criterion, device, epoch, args)
        print(f'Epoch {epoch+1} Evaluation - Test Loss: {avg_test_loss:.6f}, Landmark Error: {avg_landmark_error:.6f}')

        with open(log_filepath, 'a') as f:
            f.write(f'{epoch+1}\t{avg_test_loss:.6f}\t{avg_landmark_error:.6f}\n')

        if avg_landmark_error < best_landmark_error:
            best_landmark_error = avg_landmark_error
            best_test_loss = avg_test_loss
            save_path = os.path.join(log_dir, 'models', 'best_model.t7')
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Best model saved at {save_path} with Landmark Error: {best_landmark_error:.6f}')
        else:
            counter += 1
            print(f'EarlyStopping counter: {counter} out of {patience}')
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break

    print("Training finished.")
    print(f"Best Test Loss: {best_test_loss:.6f}, Best Landmark Error: {best_landmark_error:.6f}")

def test(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    total_error = 0
    num_batches = 0
    
    with torch.no_grad():
        for points, landmarks, _ in test_loader:
            points = points.to(device)
            landmarks = landmarks.to(device)
            
            # 추론
            points = points.permute(0, 2, 1)  # (B, N, 3) -> (B, 3, N)
            pred_heatmap = model(points)
            
            # 히트맵에서 랜드마크 위치 추출
            pred_landmarks = get_predicted_landmarks_from_heatmap(points.permute(0, 2, 1), pred_heatmap)
            
            # 손실 계산
            true_heatmap = generate_heatmap(points.permute(0, 2, 1), landmarks, sigma=2.0)
            loss = criterion(pred_heatmap, true_heatmap)
            total_loss += loss.item()
            
            # 랜드마크 오차 계산
            error = torch.mean(torch.norm(pred_landmarks - landmarks, dim=2))
            total_error += error.item()
            
            num_batches += 1
    
    return total_loss / num_batches, total_error / num_batches

def main():
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    model_path = './checkpoints/Face alignment with PAConv/custom/models/best_model.t7'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = load_model(model_path, device)
    print("Model loaded successfully!")
    
    # 테스트 데이터셋 로드
    test_dataset = FaceLandmarkData(data_dir='./dataset', partition='val')
    print(f"Loaded {len(test_dataset)} test samples")
    
    # 결과 저장 디렉토리 생성
    os.makedirs('./results', exist_ok=True)
    
    # 로그 파일 생성
    log_file = './results/training_log.txt'
    with open(log_file, 'w') as f:
        f.write("Epoch\tTest Loss\tLandmark Error\n")
    
    # 몇 개의 샘플에 대해 예측 수행
    num_samples = min(5, len(test_dataset))  # 최대 5개 샘플
    for i in range(num_samples):
        points, true_landmarks, _ = test_dataset[i]
        
        # 원본 파일 이름 가져오기
        shape_file = test_dataset.shape_files[test_dataset.indices[i]]
        landmark_file = test_dataset.landmark_files[test_dataset.indices[i]]
        shape_name = os.path.basename(shape_file)
        landmark_name = os.path.basename(landmark_file)
        
        # 예측 수행
        pred_landmarks = predict_landmarks(model, points, device)
        
        # 결과 출력
        print(f"\nSample {i+1}:")
        print(f"Shape file: {shape_name}")
        print(f"Landmark file: {landmark_name}")
        print(f"Number of points: {len(points)}")
        print(f"Number of landmarks: {len(pred_landmarks)}")
        
        # 평균 오차 계산
        error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks.numpy(), axis=1))
        print(f"Average landmark error: {error:.4f}")
        
        # 결과 저장
        result_dict = {
            'shape_file': shape_name,
            'landmark_file': landmark_name,
            'points': points.numpy(),
            'true_landmarks': true_landmarks.numpy(),
            'predicted_landmarks': pred_landmarks,
            'error': error
        }
        
        # npy 파일로 저장
        save_path = f'./results/sample_{i+1}_results.npy'
        np.save(save_path, result_dict)
        print(f"Saved results to {save_path}")
        
        # 로그 파일에 기록
        with open(log_file, 'a') as f:
            f.write(f"{i+1}\t{error:.6f}\t{error:.6f}\n")

if __name__ == "__main__":
    # Training settings
    args = parser.parse_args()
    print(f"DEBUG: args.data_dir after parsing: {args.data_dir}")
    _init_(args)  # args를 전달
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train(args)




