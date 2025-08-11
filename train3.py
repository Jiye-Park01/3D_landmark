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
import datetime

import shutil

from init import _init_, weight_init  # _init_ 함수를 명시적으로 import
from My_args import *
from augmentations import *
from dataset import FaceLandmarkData, get_dataloader
from loss import AdaptiveWingLoss
from util import main_sample, load_model, predict_landmarks, heatmap_to_coordinates
from PointTransformer_model import PointTransformerLandmark
import util


def custom_collate_fn(batch):
    points_list = [item[0] for item in batch]
    landmarks_list = [item[1] for item in batch]

    # Points are now expected to be already padded to the global max_num_points in __getitem__.
    # So we can directly stack them.
    points = torch.stack(points_list)
    
    # Stack landmarks (assuming they are already consistent in size within __getitem__)
    landmarks = torch.stack(landmarks_list)

    return points, landmarks, None

def generate_heatmap(points, landmarks, sigma=1.5):
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

    # Debugging: Check for NaNs/Infs in pred_heatmap
    if torch.isnan(pred_heatmap).any() or torch.isinf(pred_heatmap).any():
        print("DEBUG: NaN or Inf found in pred_heatmap inside get_predicted_landmarks_from_heatmap!")
        print(f"DEBUG: pred_heatmap min: {pred_heatmap.min().item()}, max: {pred_heatmap.max().item()}")
        # Optional: Save pred_heatmap for further inspection if needed
        # torch.save(pred_heatmap, "debug_pred_heatmap.pt")

    # Debugging: Print shapes
    # print(f"DEBUG: points shape: {points.shape}, pred_heatmap shape: {pred_heatmap.shape}")

    for b in range(B):
        for l in range(L):
            # Find the index of the point with the maximum heatmap value for landmark l
            max_idx = torch.argmax(pred_heatmap[b, l])
            
            # Debugging: Print max_idx and N
            # print(f"DEBUG: Batch {b}, Landmark {l}, max_idx: {max_idx.item()}, N (points count): {N}")

            # Get the 3D coordinate of this point
            pred_landmarks[b, l] = points[b, max_idx, :]
    return pred_landmarks

def evaluate(model, test_loader, criterion, device, args):
    model.eval()
    total_heatmap_loss = 0
    total_unfolding_loss = 0
    total_registration_loss = 0
    total_landmark_error = 0
    
    with torch.no_grad():
        for points, landmark, _ in test_loader:
            points, landmark = points.to(device), landmark.to(device)
            
            # 모델 예측 (히트맵) - 포인트와 랜드마크 모두 정규화
            points_normal, landmark_normal = normalize_data(points, landmark)
            points_normal = points_normal.permute(0, 2, 1)
            pred_heatmap = model(points_normal)
            
            # 정답 히트맵 생성 (정규화된 좌표계에서)
            true_heatmap = generate_heatmap(points_normal.permute(0, 2, 1), landmark_normal, sigma=args.sigma)
            
            # 1. heatmap loss
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            
            # 2. unfolding loss (local patch unfolding)
            pred_landmarks = get_predicted_landmarks_from_heatmap(points_normal.permute(0, 2, 1), pred_heatmap)
            regression_point_num = 10
            pred_unfolded = util.landmark_regression(points_normal.permute(0, 2, 1)[0], pred_heatmap[0], regression_point_num)
            gt_unfolded = util.landmark_regression(points_normal.permute(0, 2, 1)[0], true_heatmap[0], regression_point_num)
            unfolding_loss = torch.norm(pred_unfolded - gt_unfolded, dim=2).mean()
            
            # 3. registration loss (Procrustes) - 정규화된 좌표계에서
            pred_np = pred_landmarks[0].detach().cpu().numpy()
            gt_np = landmark_normal[0].detach().cpu().numpy()
            rigid = util.get_rigid(pred_np, gt_np)
            pred_reg = (np.dot(rigid[:3, :3], pred_np.T) + rigid[:3, 3:4]).T
            pred_reg = torch.from_numpy(pred_reg).to(device)
            registration_loss = torch.norm(pred_reg - landmark_normal[0], dim=1).mean()
            
            # 랜드마크 오차 계산 (정규화된 좌표계에서)
            landmark_error = torch.norm(pred_landmarks - landmark_normal, dim=2).mean()
            
            total_heatmap_loss += heatmap_loss.item()
            total_unfolding_loss += unfolding_loss.item()
            total_registration_loss += registration_loss.item()
            total_landmark_error += landmark_error.item()
    
    avg_heatmap_loss = total_heatmap_loss / len(test_loader)
    avg_unfolding_loss = total_unfolding_loss / len(test_loader)
    avg_registration_loss = total_registration_loss / len(test_loader)
    avg_landmark_error = total_landmark_error / len(test_loader)
    
    # 가중치 적용된 전체 loss 계산
    avg_total_loss = 1.0 * avg_heatmap_loss + 0.5 * avg_unfolding_loss + 0.3 * avg_registration_loss
    
    return avg_total_loss, avg_landmark_error, avg_heatmap_loss, avg_unfolding_loss, avg_registration_loss

def train(args):
    _init_(args) # Initialize folders
    
    import os # Add os import for getcwd
    print(f"DEBUG: Current working directory: {os.getcwd()}")

    # print(f"DEBUG: args.data_dir after _init_: {args.data_dir}")

    # # Convert absolute data_dir to relative if necessary
    # # The previous parsing seems to incorrectly make it absolute, force it to be relative to CWD
    # args.data_dir = './dataset'

    # print(f"Number of points to sample: {args.num_points}")
    # print(f"Number of landmarks: {args.num_landmarks}") # args.num_landmarks is 57

    # Freeze 단계에서는 A.npy 파일들만 사용, unfreeze 단계에서는 모든 파일 사용
    train_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='train', use_only_a_files=True)
    test_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='val', use_only_a_files=True)
    
    print(f"Initial dataset size (A.npy only) - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Check dataset size
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: Dataset is empty after splitting. Check data_dir and file patterns.")
        return

    device = torch.device("cuda" if args.cuda else "cpu") # Move device definition up
    print(f"Using device: {device}")
        
    # Add normalized landmark ranges to args, converted to tensor and moved to device
    # These ranges are calculated from the entire dataset during FaceLandmarkData.__init__
    args.landmark_ranges_min_norm = torch.from_numpy(train_dataset.landmark_ranges_min).float().to(device)
    args.landmark_ranges_max_norm = torch.from_numpy(train_dataset.landmark_ranges_max).float().to(device)

    print(f"DEBUG: Normalized landmark ranges min shape: {args.landmark_ranges_min_norm.shape}")
    print(f"DEBUG: Normalized landmark ranges max shape: {args.landmark_ranges_max_norm.shape}")
        
    # Use custom_collate_fn to handle None values from dataset (if any)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=custom_collate_fn)

    # Load pretrained Point Transformer weights
    pretrained_path = args.pretrained_path
    
    # Initialize model with pretrained weights
    model = PointTransformerLandmark(args, args.num_landmarks, pretrained_path=pretrained_path).to(device)
    
    # Freeze → Unfreeze 전략을 위한 설정
    freeze_epochs = args.freeze_epochs
    unfreeze_epoch = args.unfreeze_epoch
    
    # Optimizer (Initialized once here)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    best_test_loss = float('inf')
    best_landmark_error = float('inf')

    # Early Stopping parameters
    patience_limit = 10 # Number of epochs to wait for improvement
    min_delta = 1e-4
    patience_counter = 0
    start_epoch = 0 # Initialize start_epoch

    # freeze 단계에서 sigma=1.5로 설정
    args.sigma = 1.5

    # 로그 파일명에 타임스탬프 추가
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('./checkpoints', args.exp_name, args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    
    # Load pretrained model if model_path is provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"DEBUG: Attempting to load model from: {os.path.abspath(args.model_path)}")
        print(f"Loading pretrained model from {args.model_path}")
        try:
            checkpoint = torch.load(args.model_path, map_location=device)
            print(f"DEBUG: Type of loaded checkpoint: {type(checkpoint)}")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_test_loss = checkpoint['best_test_loss']
            best_landmark_error = checkpoint['best_landmark_error']
            patience_counter = checkpoint['patience_counter']
            print(f"Model and training state loaded successfully. Resuming from epoch {start_epoch}.")
            print(f"Current best test loss: {best_test_loss:.6f}, best landmark error: {best_landmark_error:.6f}")
        except Exception as e:
            print(f"Error loading checkpoint from {args.model_path}: {e}")
            print("Proceeding with random initialization and fresh training state.")
            model.apply(weight_init) # Apply weight initialization if loading fails
            # Keep initial best_test_loss, best_landmark_error, patience_counter as inf/0
    else:
        print("No pretrained model path provided or file not found. Initializing weights randomly and starting fresh.")
        model.apply(weight_init)

    # DataParallel if multiple GPUs
    if args.cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Loss function (Heatmap loss)
    criterion = AdaptiveWingLoss(omega=14, theta=0.5, epsilon=1, alpha=2.1)

    # Learning rate scheduler (optional)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) # Example scheduler

    # Write header if file is new or starting fresh
    if not os.path.exists(log_filepath) or os.stat(log_filepath).st_size == 0 or start_epoch == 0:
        with open(log_filepath, 'a') as f:
            f.write('Epoch\tTest Loss\tLandmark Error\tHeatmap Loss\tUnfolding Loss\tRegistration Loss\tTotal Loss\n')

    print("Start training...")
    
    for epoch in range(start_epoch, args.epochs): # Use start_epoch here
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Freeze → Unfreeze 전략 구현
        if epoch == unfreeze_epoch:
            print("Unfreezing backbone encoder for fine-tuning...")
            (model.module if isinstance(model, torch.nn.DataParallel) else model).unfreeze_backbone()

            patience_counter = 0
            best_test_loss = float('inf')
            best_landmark_error = float('inf')
            print("Early stopping status reset after unfreeze.")
            
            # Unfreeze 단계에서 sigma=1.0으로 변경
            args.sigma = 1.0
            
            # Switch to A files only for unfreeze phase as well
            print("(Unfreeze) Still using only A file types (A_vertices/landmarks) for training and validation...")
            train_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='train', use_only_a_files=True)
            test_dataset = FaceLandmarkData(data_dir=args.data_dir, num_points=args.num_points, partition='val', use_only_a_files=True)
            
            # Recreate data loaders with new datasets
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, collate_fn=custom_collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=custom_collate_fn, drop_last=True)
            
            print(f"(Unfreeze) Dataset size (A only) - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
            # Reinitialize optimizer with all parameters
            optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        
        model.train()
        total_train_loss = 0


        for i, data in enumerate(train_loader):


            points, landmark, _ = data
            points, landmark = points.to(device), landmark.to(device)
            optimizer.zero_grad()

            # 모델 예측 (히트맵) - 포인트와 랜드마크 모두 정규화
            points_normal, landmark_normal = normalize_data(points, landmark)
            points_normal = points_normal.permute(0, 2, 1)
            pred_heatmap = model(points_normal)

            # 정답 히트맵 생성 (정규화된 좌표계에서)
            true_heatmap = generate_heatmap(points_normal.permute(0, 2, 1), landmark_normal, sigma=args.sigma)

            # 1. heatmap loss
            heatmap_loss = criterion(pred_heatmap, true_heatmap)

            # 2. unfolding loss (local patch unfolding)
            # 예측 랜드마크 좌표 (정규화된 좌표계에서)
            pred_landmarks = get_predicted_landmarks_from_heatmap(points_normal.permute(0, 2, 1), pred_heatmap)  # (B, L, 3)
            # unfolding: landmark_regression 함수 활용 (regression_point_num=10 예시)
            regression_point_num = 10
            pred_unfolded = util.landmark_regression(points_normal.permute(0, 2, 1)[0], pred_heatmap[0], regression_point_num)  # (1, L, 3)
            gt_unfolded = util.landmark_regression(points_normal.permute(0, 2, 1)[0], true_heatmap[0], regression_point_num)    # (1, L, 3)
            unfolding_loss = torch.norm(pred_unfolded - gt_unfolded, dim=2).mean()

            # 3. registration loss (Procrustes) - 정규화된 좌표계에서
            # registration: get_rigid 함수 활용
            pred_np = pred_landmarks[0].detach().cpu().numpy()  # (L, 3)
            gt_np = landmark_normal[0].detach().cpu().numpy()          # (L, 3)
            rigid = util.get_rigid(pred_np, gt_np)     # (3, 4)
            # pred를 rigid 변환
            pred_reg = (np.dot(rigid[:3, :3], pred_np.T) + rigid[:3, 3:4]).T
            pred_reg = torch.from_numpy(pred_reg).to(device)
            registration_loss = torch.norm(pred_reg - landmark_normal[0], dim=1).mean()

            # 가중치 적용된 총 손실
            total_loss = 1.0 * heatmap_loss + 0.5 * unfolding_loss + 0.3 * registration_loss

            total_train_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()
            
            # Print training loss periodically
            if (i + 1) % 10 == 0:
                print(f'  Iter {i+1}/{len(train_loader)}, Loss: {total_loss.item():.6f}, Heatmap Loss: {heatmap_loss.item():.6f}, Unfolding Loss: {unfolding_loss.item():.6f}, Registration Loss: {registration_loss.item():.6f}')

        # scheduler.step() # If using learning rate scheduler

        # Evaluation after each epoch
        avg_test_loss, avg_landmark_error, avg_heatmap_loss, avg_unfolding_loss, avg_registration_loss = evaluate(model, test_loader, criterion, device, args)
        print(f'Epoch {epoch+1} Evaluation - Test Loss: {avg_test_loss:.6f}, Landmark Error: {avg_landmark_error:.6f}')

        # Save training log
        with open(log_filepath, 'a') as f:
            f.write(f'{epoch+1}\t{avg_test_loss:.6f}\t{avg_landmark_error:.6f}\t{avg_heatmap_loss:.6f}\t{avg_unfolding_loss:.6f}\t{avg_registration_loss:.6f}\t{avg_test_loss:.6f}\n')

        # Early Stopping logic - 전체 loss 기준으로 변경
        if avg_test_loss < best_test_loss - 0.0001:
            print(f'Total loss improved from {best_test_loss:.6f} to {avg_test_loss:.6f}. Saving model.')
            best_test_loss = avg_test_loss
            best_landmark_error = avg_landmark_error # Save corresponding landmark error
            patience_counter = 0 # Reset patience counter
            
            os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
            save_path = os.path.join(log_dir, 'models', 'best_model.t7')
            print(f'DEBUG: Attempting to save model to absolute path: {os.path.abspath(save_path)}')
            # Save model state_dict (handle DataParallel if used)
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            state = {
                'epoch': epoch + 1, # Save next epoch number
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_loss': best_test_loss,
                'best_landmark_error': best_landmark_error,
                'patience_counter': patience_counter,
            }
            torch.save(state, save_path)
            print(f'Best model and training state saved at {os.path.abspath(save_path)} with Total Loss: {best_test_loss:.6f}')
        else:
            patience_counter += 1
            print(f'No improvement. Early stopping patience: {patience_counter}/{patience_limit}')
            if patience_counter >= patience_limit:
                print('Early stopping triggered!')
                break # Exit the training loop

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
            
            # 정규화 - 포인트와 랜드마크 모두
            points_normal, landmarks_normal = normalize_data(points, landmarks)
            points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 추론
            pred_heatmap = model(points_normal)
            
            # 히트맵에서 랜드마크 위치 추출
            B, L, N = pred_heatmap.shape
            pred_landmarks = torch.zeros(B, L, 3).to(device)
            for b in range(B):
                for l in range(L):
                    max_idx = torch.argmax(pred_heatmap[b, l])
                    pred_landmarks[b, l] = points_normal[b, :, max_idx]
            
            # 손실 계산 (정규화된 좌표계에서)
            loss = criterion(pred_heatmap, landmarks_normal)
            total_loss += loss.item()
            
            # 랜드마크 오차 계산 (정규화된 좌표계에서)
            error = torch.mean(torch.norm(pred_landmarks - landmarks_normal, dim=2))
            total_error += error.item()
            
            num_batches += 1
    
    return total_loss / num_batches, total_error / num_batches

def main(args):
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    model_path = '../3D_pointtransformer/autoencoder_pointTransformer/pointtransformer_autoencoder/model/model_best.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = load_model(model_path, device, args)
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
        points, true_landmarks = test_dataset[i]
        
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

    args.num_landmarks = 68

    train(args)
    main(args)




