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
import matplotlib.pyplot as plt

from init import _init_, weight_init  # _init_ 함수를 명시적으로 import
from My_args import *
from augmentations import *
from dataset import FaceLandmarkData, custom_collate_fn
from loss import AdaptiveWingLoss
from util import main_sample
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    points = torch.stack([item[0] for item in batch])
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
    dist_sq = torch.sum((points - landmarks)**2, dim=-1)
    
    # Calculate heatmap using Gaussian kernel
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    
    # Optional: Normalize heatmap to [0, 1] range for better visualization, but not for loss calculation
    # min_val = torch.min(heatmap, dim=2, keepdim=True)[0]
    # max_val = torch.max(heatmap, dim=2, keepdim=True)[0]
    # if max_val - min_val > 1e-6:
    #     heatmap = (heatmap - min_val) / (max_val - min_val)
    # else:
    #     heatmap = torch.zeros_like(heatmap)

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

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_landmark_error = 0
    
    with torch.no_grad():
        for data in test_loader:
            points, landmark, _ = data
            points, landmark = points.to(device), landmark.to(device)
            
            # 정규화된 포인트와 랜드마크를 함께 얻음
            points_normal, landmark_normal = normalize_data(points, landmark)
            
            # 모델 입력 형태에 맞게 차원 변경 (B, 3, N)
            points_model_input = points_normal.permute(0, 2, 1)
            
            # 모델 예측 (히트맵)
            pred_heatmap = model(points_model_input)
            
            # 정답 히트맵 생성 (정규화된 랜드마크 사용)
            # generate_heatmap은 (B, N, 3) 형태의 포인트를 기대합니다.
            true_heatmap = generate_heatmap(points_normal, landmark_normal, sigma=2.0)
            
            # 히트맵 손실 계산
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            total_loss += heatmap_loss.item()

            # 예측 랜드마크 위치 추출
            # get_predicted_landmarks_from_heatmap는 (B, N, 3) 형태의 포인트를 기대합니다.
            pred_landmarks = get_predicted_landmarks_from_heatmap(points_normal, pred_heatmap)

            # 랜드마크 오차 계산 (평균 거리)
            landmark_error = torch.norm(pred_landmarks - landmark_normal, dim=2).mean()
            total_landmark_error += landmark_error.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_landmark_error = total_landmark_error / len(test_loader)
    
    return avg_loss, avg_landmark_error

def save_heatmap_as_obj(points, heatmap_values, landmark_idx, filename):
    """
    Saves point cloud with heatmap values as vertex colors to an OBJ file.
    Args:
        points: (N, 3) numpy array of point coordinates.
        heatmap_values: (N,) numpy array of heatmap values for a specific landmark.
        landmark_idx: Index of the landmark (for filename).
        filename: Path to save the OBJ file.
    """
    num_points = points.shape[0]

    # Normalize heatmap values to [0, 1] for colormap mapping
    min_val = np.min(heatmap_values)
    max_val = np.max(heatmap_values)
    if max_val - min_val > 1e-6:
        normalized_heatmap = (heatmap_values - min_val) / (max_val - min_val)
    else:
        normalized_heatmap = np.zeros_like(heatmap_values)

    # Map normalized heatmap values to colors (simple DarkRed to Yellow mapping)
    colors = np.zeros((num_points, 3))
    # Interpolate from DarkRed (0.2, 0, 0) to Yellow (1, 1, 0)
    colors[:, 0] = 0.2 + 0.8 * normalized_heatmap # R
    colors[:, 1] = normalized_heatmap # G
    colors[:, 2] = 0 # B

    with open(filename, 'w') as f:
        # Write vertices with colors
        for i in range(num_points):
            f.write(f'v {points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} {colors[i, 0]:.6f} {colors[i, 1]:.6f} {colors[i, 2]:.6f}\n')

    print(f"Heatmap for Landmark {landmark_idx} saved to {filename}")

def visualize_heatmap(points, landmarks, heatmap, save_path=None):
    """
    Visualize points, landmarks and heatmap in 3D
    Args:
        points: (N, 3) numpy array of point cloud
        landmarks: (L, 3) numpy array of landmark positions
        heatmap: (L, N) numpy array of heatmap values
        save_path: path to save the visualization (optional)
    """
    fig = plt.figure(figsize=(18, 6))

    # Plot 1: Original points and landmarks
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.5)
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='red', s=50)
    ax1.set_title('Points and Landmarks')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.view_init(elev=90, azim=-90) # Z-up, looking from the front-ish
    ax1.set_box_aspect([np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])]) # Equal aspect ratio

    # Plot 2: Heatmap for first landmark (index 0)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=heatmap[0], cmap='hot', s=1)
    ax2.scatter(landmarks[0, 0], landmarks[0, 1], landmarks[0, 2],
                c='red', s=100, marker='*')
    ax2.set_title(f'Heatmap for Landmark 0')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.view_init(elev=90, azim=-90) # Z-up, looking from the front-ish
    ax2.set_box_aspect([np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])]) # Equal aspect ratio
    fig.colorbar(scatter, ax=ax2, pad=0.1)

    # Plot 3: Heatmap for middle landmark
    mid_idx = len(landmarks) // 2
    ax3 = fig.add_subplot(133, projection='3d')
    scatter = ax3.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=heatmap[mid_idx], cmap='hot', s=1)
    ax3.scatter(landmarks[mid_idx, 0], landmarks[mid_idx, 1], landmarks[mid_idx, 2],
                c='red', s=100, marker='*')
    ax3.set_title(f'Heatmap for Landmark {mid_idx}')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.view_init(elev=90, azim=-90) # Z-up, looking from the front-ish
    ax3.set_box_aspect([np.ptp(points[:,0]), np.ptp(points[:,1]), np.ptp(points[:,2])]) # Equal aspect ratio
    fig.colorbar(scatter, ax=ax3, pad=0.1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

def visualize_heatmap_comparison(points_normal, landmark_normal, true_heatmap, pred_heatmap, epoch, save_dir='./visualizations/comparisons'):
    """
    Visualize and compare ground truth and predicted heatmaps for a sample.
    Args:
        points_normal: (B, N, 3) numpy array of normalized points.
        landmark_normal: (B, L, 3) numpy array of normalized landmarks.
        true_heatmap: (B, L, N) numpy array of ground truth heatmaps.
        pred_heatmap: (B, L, N) numpy array of predicted heatmaps.
        epoch: Current epoch number.
        save_dir: Directory to save comparison images.
    """
    # Select the first sample in the batch (inputs are already numpy arrays)
    points_sample = points_normal[0]
    landmark_sample = landmark_normal[0]
    true_heatmap_sample = true_heatmap[0]
    pred_heatmap_sample = pred_heatmap[0]

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Select a few landmarks to visualize (e.g., first, middle, last)
    landmark_indices_to_plot = [0, len(landmark_sample) // 2, len(landmark_sample) - 1]
    
    for lm_idx in landmark_indices_to_plot:
        fig = plt.figure(figsize=(12, 6))

        # Plot 1: Ground Truth Heatmap
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                               c=true_heatmap_sample[lm_idx], cmap='hot', s=1)
        ax1.scatter(landmark_sample[lm_idx, 0], landmark_sample[lm_idx, 1], landmark_sample[lm_idx, 2],
                    c='red', s=100, marker='*')
        ax1.set_title(f'Epoch {epoch} - GT Heatmap for Landmark {lm_idx}')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax1.view_init(elev=90, azim=-90) # Z-up, looking from the front-ish
        ax1.set_box_aspect([np.ptp(points_sample[:,0]), np.ptp(points_sample[:,1]), np.ptp(points_sample[:,2])]) # Equal aspect ratio
        fig.colorbar(scatter1, ax=ax1, pad=0.1)

        # Plot 2: Predicted Heatmap
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2],
                               c=pred_heatmap_sample[lm_idx], cmap='hot', s=1)
        ax2.scatter(landmark_sample[lm_idx, 0], landmark_sample[lm_idx, 1], landmark_sample[lm_idx, 2],
                    c='red', s=100, marker='*')
        ax2.set_title(f'Epoch {epoch} - Pred Heatmap for Landmark {lm_idx}')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        ax2.view_init(elev=90, azim=-90) # Z-up, looking from the front-ish
        ax2.set_box_aspect([np.ptp(points_sample[:,0]), np.ptp(points_sample[:,1]), np.ptp(points_sample[:,2])]) # Equal aspect ratio
        fig.colorbar(scatter2, ax=ax2, pad=0.1)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'epoch_{epoch}_landmark_{lm_idx}_comparison.png')
        plt.savefig(save_path)
        plt.close(fig) # Close the figure to free memory

    print(f"Comparison visualizations saved for epoch {epoch} to {save_dir}")

def train(args):
    _init_(args) # Initialize folders
    
    print(f"DEBUG: args.data_dir after _init_: {args.data_dir}")

    # Convert absolute data_dir to relative if necessary
    # The previous parsing seems to incorrectly make it absolute, force it to be relative to CWD
    args.data_dir = './dataset'

    print(f"Number of points to sample: {args.num_points}")
    print(f"Number of landmarks: {args.num_landmarks}") # args.num_landmarks is 57

    train_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='train')
    test_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='val')

    # Check dataset size
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Dataset is empty after splitting. Check data_dir and file patterns.")
        return
        
    # Use custom_collate_fn to handle None values from dataset (if any)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=custom_collate_fn)

    device = torch.device("cuda" if args.cuda else "cpu")
    print(f"Using device: {device}")

    model = PAConv(args, args.num_landmarks).to(device)
    # Apply weight initialization
    model.apply(weight_init)

    # DataParallel if multiple GPUs
    if args.cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    # Loss function (Heatmap loss)
    criterion = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1, alpha=2.1)

    # Learning rate scheduler (optional)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # Example scheduler

    best_test_loss = float('inf')
    best_landmark_error = float('inf')

    # Log file setup
    log_dir = os.path.join('./checkpoints', args.exp_name, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'training_log.txt')
    
    # Write header if file is new
    if not os.path.exists(log_filepath) or os.stat(log_filepath).st_size == 0:
        with open(log_filepath, 'w') as f:
            f.write('Epoch\tTest Loss\tLandmark Error\n')

    print("Start training...")
    
    # Create directory for visualizations if it doesn't exist
    os.makedirs('./visualizations', exist_ok=True)

    # --- Heatmap Visualization for the first sample ---
    try:
        points_sample, landmarks_sample, _ = train_dataset[0]
        points_sample, landmarks_sample = points_sample.to(device).unsqueeze(0), landmarks_sample.to(device).unsqueeze(0) # Add batch dim and move to device

        # 정규화된 포인트와 랜드마크를 함께 얻음
        points_normal_sample, landmark_normal_sample = normalize_data(points_sample, landmarks_sample)

        # 모델 입력 형태에 맞게 차원 변경 (B, 3, N)
        points_model_input_sample = points_normal_sample.permute(0, 2, 1)

        # 모델 예측 (히트맵) - 시각화를 위해 모델 예측도 필요
        with torch.no_grad():
            pred_heatmap_sample = model(points_model_input_sample)
        
        # 정답 히트맵 생성 (정규화된 랜드마크 사용)
        # generate_heatmap은 (B, N, 3) 형태의 포인트를 기대합니다.
        true_heatmap_sample = generate_heatmap(points_normal_sample, landmark_normal_sample, sigma=2.0)

        # convert tensors to numpy and remove batch dim for visualization function
        points_np = points_normal_sample.squeeze(0).cpu().numpy()
        landmarks_np = landmark_normal_sample.squeeze(0).cpu().numpy()
        true_heatmap_np = true_heatmap_sample.squeeze(0).cpu().numpy()
        pred_heatmap_np = pred_heatmap_sample.squeeze(0).cpu().numpy()

        # Visualize the heatmap
        visualize_heatmap_comparison(points_np, landmarks_np, true_heatmap_np, pred_heatmap_np,
                                     0, # Use 0 for epoch since this is before training starts
                                     save_dir='./visualizations') # Save to main visualizations folder for initial check

    except IndexError:
        print("Warning: train_dataset is empty, skipping heatmap visualization.")
    except Exception as e:
        print(f"An error occurred during heatmap visualization: {e}")
    # --------------------------------------------------

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        model.train()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader):
            points, landmark, _ = data
            points, landmark = points.to(device), landmark.to(device)
            
            optimizer.zero_grad()

            # 정규화된 포인트와 랜드마크를 함께 얻음
            points_normal, landmark_normal = normalize_data(points, landmark)

            # 모델 입력 형태에 맞게 차원 변경 (B, 3, N)
            points_model_input = points_normal.permute(0, 2, 1)

            # 모델 예측 (히트맵)
            pred_heatmap = model(points_model_input)
            
            # 정답 히트맵 생성 (정규화된 랜드마크 사용)
            # generate_heatmap은 (B, N, 3) 형태의 포인트를 기대합니다.
            true_heatmap = generate_heatmap(points_normal, landmark_normal, sigma=args.sigma)
            
            # Heatmap 손실 계산
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            
            # 예측 랜드마크 위치 추출
            # get_predicted_landmarks_from_heatmap는 (B, N, 3) 형태의 포인트를 기대합니다.
            pred_landmarks = get_predicted_landmarks_from_heatmap(points_normal, pred_heatmap)

            # 랜드마크 위치 오차 계산 (Position Loss)
            # Use L2 norm (Euclidean distance) as position loss
            position_loss = torch.norm(pred_landmarks - landmark_normal, dim=2).mean() # Average error per landmark per sample, then average over batch

            # 총 손실 계산 (Heatmap Loss + alpha * Position Loss)
            alpha = 0.1 # Weight for position loss
            total_loss = heatmap_loss + alpha * position_loss

            total_train_loss += total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            # Print training loss periodically
            if (i + 1) % 10 == 0:
                print(f'  Iter {i+1}/{len(train_loader)}, Loss: {total_loss.item():.6f}')
                
        # Visualize heatmap comparison periodically (e.g., every 10 epochs)
        if (epoch + 1) % 10 == 0:
             with torch.no_grad(): # Ensure no gradients are computed during visualization
                 try:
                     data_sample = next(iter(train_loader))
                     points_sample, landmark_sample, _ = data_sample
                     points_sample, landmark_sample = points_sample.to(device), landmark_sample.to(device)
                     
                     # 정규화된 포인트와 랜드마크를 함께 얻음
                     points_normal_sample, landmark_normal_sample = normalize_data(points_sample, landmark_sample)
                     
                     # 모델 입력 형태에 맞게 차원 변경 (B, 3, N)
                     points_model_input_sample = points_normal_sample.permute(0, 2, 1)
                     
                     # 모델 예측
                     pred_heatmap_sample = model(points_model_input_sample)
                     # 정답 히트맵 생성 (정규화된 랜드마크 사용)
                     # generate_heatmap은 (B, N, 3) 형태의 포인트를 기대합니다.
                     true_heatmap_sample = generate_heatmap(points_normal_sample, landmark_normal_sample, sigma=args.sigma)

                     # convert tensors to numpy and remove batch dim for visualization function
                     points_np = points_normal_sample.squeeze(0).cpu().numpy()
                     landmarks_np = landmark_normal_sample.squeeze(0).cpu().numpy()
                     true_heatmap_np = true_heatmap_sample.squeeze(0).cpu().numpy()
                     pred_heatmap_np = pred_heatmap_sample.squeeze(0).cpu().numpy()

                     visualize_heatmap_comparison(points_np, landmarks_np, 
                                                  true_heatmap_np, pred_heatmap_np, 
                                                  epoch + 1) # Pass epoch + 1 for correct naming
                 except Exception as e:
                     print(f"Warning: Could not visualize heatmap comparison for epoch {epoch + 1} - {e}")

        # scheduler.step() # If using learning rate scheduler

        # Evaluation after each epoch
        avg_test_loss, avg_landmark_error = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1} Evaluation - Test Loss: {avg_test_loss:.6f}, Landmark Error: {avg_landmark_error:.6f}')

        # Save training log
        with open(log_filepath, 'a') as f:
            f.write(f'{epoch+1}\t{avg_test_loss:.6f}\t{avg_landmark_error:.6f}\n')

        # Save best model based on Landmark Error
        if avg_landmark_error < best_landmark_error:
            best_landmark_error = avg_landmark_error
            best_test_loss = avg_test_loss # Save corresponding test loss
            save_path = os.path.join(log_dir, 'models', 'best_model.t7')
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            # Save model state_dict (handle DataParallel if used)
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Best model saved at {save_path} with Landmark Error: {best_landmark_error:.6f}')

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
            
            # 정규화
            point_normal = normalize_data(points)
            point_normal = point_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 추론
            pred_heatmap = model(point_normal)
            
            # 히트맵에서 랜드마크 위치 추출
            B, L, N = pred_heatmap.shape
            pred_landmarks = torch.zeros(B, L, 3).to(device)
            for b in range(B):
                for l in range(L):
                    max_idx = torch.argmax(pred_heatmap[b, l])
                    pred_landmarks[b, l] = point_normal[b, :, max_idx]
            
            # 손실 계산
            loss = criterion(pred_heatmap, landmarks)
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
    main()




