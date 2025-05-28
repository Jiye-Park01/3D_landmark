# '''
# @Author: Yuan Wang
# @Contact: wangyuan2020@ia.ac.cn
# @File: train.py
# @Time: 2021/12/02 09:59 AM
# '''

# import os
# import math
# import time
# import numpy as np
# from scipy import io
# import torch.optim as optim
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
# import torch
# import torch.nn as nn

# from init import _init_, weight_init  # _init_ 함수를 명시적으로 import
# from My_args import *
# from augmentations import *
# from dataset import FaceLandmarkData
# from loss import AdaptiveWingLoss
# from util import main_sample
# from PAConv_model import PAConv

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def custom_collate_fn(batch):
#     points = torch.stack([item[0] for item in batch])
#     landmarks = torch.stack([item[1] for item in batch])
#     return points, landmarks, None

# def generate_heatmap(points, landmarks, sigma=3):
#     """히트맵 생성 함수"""
#     B, C, N = points.shape
#     heatmap = torch.zeros(B, landmarks.shape[1], N).to(points.device)
    
#     for b in range(B):
#         for l in range(landmarks.shape[1]):
#             landmark = landmarks[b, l]  # (3,)
#             # 각 포인트와 랜드마크 사이의 거리 계산
#             diff = points[b].permute(1, 0) - landmark  # (N, 3)
#             dist = torch.sum(diff * diff, dim=1)  # (N,)
#             # 가우시안 히트맵 생성
#             heatmap[b, l] = torch.exp(-dist / (2 * sigma * sigma))
    
#     return heatmap

# def evaluate(model, test_loader, criterion, device):
#     model.eval()
#     total_loss = 0
#     total_samples = 0
#     total_landmark_error = 0
    
#     with torch.no_grad():
#         for point, landmark, _ in test_loader:
#             if torch.cuda.is_available():
#                 point = point.to(device)
#                 landmark = landmark.to(device)
            
#             point_normal = normalize_data(point)
#             point_normal = point_normal.permute(0, 2, 1)
#             pred_heatmap = model(point_normal)
            
#             # Compute the loss function
#             heatmap = generate_heatmap(point_normal, landmark, sigma=args.sigma)
#             loss = criterion(pred_heatmap, heatmap)
            
#             # Convert heatmap to landmark positions
#             B, L, N = pred_heatmap.shape
#             pred_landmarks = torch.zeros(B, L, 3).to(device)
#             for b in range(B):
#                 for l in range(L):
#                     # Get the point with maximum heatmap value
#                     max_idx = torch.argmax(pred_heatmap[b, l])
#                     pred_landmarks[b, l] = point_normal[b, :, max_idx]
            
#             # Calculate landmark error (Euclidean distance)
#             landmark_error = torch.norm(pred_landmarks - landmark, dim=2).mean()
            
#             total_loss += loss.item()
#             total_landmark_error += landmark_error.item()
#             total_samples += 1
    
#     avg_loss = total_loss / total_samples
#     avg_landmark_error = total_landmark_error / total_samples
#     return avg_loss, avg_landmark_error

# def train(args):
#     writer = SummaryWriter('runs/3D_face_alignment')
    
#     # 로그 파일 생성
#     log_dir = os.path.join('./checkpoints', args.exp_name, args.dataset)
#     os.makedirs(log_dir, exist_ok=True)
#     log_file = os.path.join(log_dir, 'training_log.txt')
#     with open(log_file, 'w') as f:
#         f.write("Epoch\tTest Loss\tLandmark Error\n")
    
#     # Convert absolute path to relative path if needed
#     if args.data_dir.startswith('/'):
#         args.data_dir = '.' + args.data_dir
    
#     # Dataset Random partition
#     FaceLandmark = FaceLandmarkData(data_dir=args.data_dir, partition='trainval')
#     print(f"Total dataset size: {len(FaceLandmark)}")
    
#     # 데이터셋이 비어있는지 확인
#     if len(FaceLandmark) == 0:
#         raise ValueError("Dataset is empty! Please check your data directory and file patterns.")
    
#     # 9:1 비율로 train과 test 분할
#     train_size = int(len(FaceLandmark) * 0.9)
#     test_size = len(FaceLandmark) - train_size
#     print(f"Train size: {train_size}, Test size: {test_size}")
    
#     torch.manual_seed(args.dataset_seed)
    
#     # Prepare the dataset and dataloader 
#     train_dataset, test_dataset = torch.utils.data.random_split(FaceLandmark, [train_size, test_size])
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=custom_collate_fn)
#     test_loader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    
#     # data argument
#     ScaleAndTranslate = PointcloudScaleAndTranslate()
#     MOMENTUM_ORIGINAL = 0.1
#     MOMENTUM_DECCAY = 0.5

#     # select a model to train
#     model = PAConv(args, 57).to(device)   # 57 landmarks
#     model.apply(weight_init)
#     model = nn.DataParallel(model)

#     print('let us use', torch.cuda.device_count(), 'GPUs')
#     if args.loss == 'adaptive_wing':
#         criterion = AdaptiveWingLoss()
#     elif args.loss == 'mse':
#         criterion = nn.MSELoss()
#     if args.use_sgd:
#         print("Use SGD")
#         opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=args.weight_decay)
#     else:
#         print("Use Adam")
#         opt = optim.Adam(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.weight_decay)
#     if args.scheduler == 'cos':
#         scheduler = CosineAnnealingLR(opt, T_max=100, eta_min=0.0001)
#     elif args.scheduler == 'step':
#         scheduler = StepLR(opt, step_size=40, gamma=0.9)

#     best_test_loss = float('inf')
#     loss_epoch = 0.0
#     for epoch in range(args.epochs):
#         iters = 0
#         model.train()
#         for point, landmark, _ in train_loader:  # seg는 None이므로 _로 받음
#             iters = iters + 1
#             if args.no_cuda == False:
#                 point = point.to(device)                   # point: (1, N, 3) where N is variable
#                 landmark = landmark.to(device)             # landmark: (1, 57, 3)
            
#             point_normal = normalize_data(point)           # point_normal: (1, N, 3)
#             point_normal = ScaleAndTranslate(point_normal)
#             opt.zero_grad()
#             point_normal = point_normal.permute(0, 2, 1)   # point: (1, 3, N)
#             pred_heatmap = model(point_normal)

#             # Compute the loss function 
#             heatmap = generate_heatmap(point_normal, landmark, sigma=args.sigma)
#             loss = criterion(pred_heatmap, heatmap)
#             loss.backward()
#             loss_epoch = loss_epoch + loss
#             opt.step()
            
#             if iters % 10 == 0:  # 10번의 iteration마다 한 번씩만 출력
#                 print('Epoch: [%d / %d] Train_Iter: [%d /%d] loss: %.4f' % (epoch + 1, args.epochs, iters, len(train_loader), loss.item()))
        
#         # Evaluate on test set
#         test_loss, test_landmark_error = evaluate(model, test_loader, criterion, device)
#         print('Epoch: [%d / %d] Test Loss: %.4f, Landmark Error: %.4f' % (epoch + 1, args.epochs, test_loss, test_landmark_error))
#         writer.add_scalar('Test_Loss', test_loss, epoch + 1)
#         writer.add_scalar('Test_Landmark_Error', test_landmark_error, epoch + 1)
        
#         # Save best model
#         if test_loss < best_test_loss:
#             best_test_loss = test_loss
#             os.makedirs('./checkpoints/%s/%s/models/' % (args.exp_name, args.dataset), exist_ok=True)
#             torch.save(model.state_dict(), './checkpoints/%s/%s/models/best_model.t7' % (args.exp_name, args.dataset))
#             print(f'New best model saved! Test Loss: {test_loss:.4f}, Landmark Error: {test_landmark_error:.4f}')
        
#         # Save checkpoint every 5 epochs
#         if (epoch + 1) % 5 == 0:
#             os.makedirs('./checkpoints/%s/%s/models/' % (args.exp_name, args.dataset), exist_ok=True)
#             torch.save(model.state_dict(), './checkpoints/%s/%s/models/model_epoch_%d.t7' % (args.exp_name, args.dataset, epoch+1))
        
#         if args.scheduler == 'cos':
#             scheduler.step()
#         elif args.scheduler == 'step':
#             if opt.param_groups[0]['lr'] > 1e-5:
#                 scheduler.step()
#             if opt.param_groups[0]['lr'] < 1e-5:
#                 for param_group in opt.param_groups:
#                     param_group['lr'] = 1e-5
#         writer.add_scalar('Train_Loss', loss_epoch / ((epoch + 1) * len(train_loader)), epoch + 1)

#         # 로그 파일에 기록
#         with open(log_file, 'a') as f:
#             f.write(f"{epoch+1}\t{test_loss:.6f}\t{test_landmark_error:.6f}\n")

# def test(test_loader, model, criterion, device):
#     model.eval()
#     total_loss = 0
#     total_error = 0
#     num_batches = 0
    
#     with torch.no_grad():
#         for points, landmarks, _ in test_loader:
#             points = points.to(device)
#             landmarks = landmarks.to(device)
            
#             # 정규화
#             point_normal = normalize_data(points)
#             point_normal = point_normal.permute(0, 2, 1)  # (B, 3, N)
            
#             # 추론
#             pred_heatmap = model(point_normal)
            
#             # 히트맵에서 랜드마크 위치 추출
#             B, L, N = pred_heatmap.shape
#             pred_landmarks = torch.zeros(B, L, 3).to(device)
#             for b in range(B):
#                 for l in range(L):
#                     max_idx = torch.argmax(pred_heatmap[b, l])
#                     pred_landmarks[b, l] = point_normal[b, :, max_idx]
            
#             # 손실 계산
#             loss = criterion(pred_heatmap, landmarks)
#             total_loss += loss.item()
            
#             # 랜드마크 오차 계산
#             error = torch.mean(torch.norm(pred_landmarks - landmarks, dim=2))
#             total_error += error.item()
            
#             num_batches += 1
    
#     return total_loss / num_batches, total_error / num_batches

# def main():
#     # GPU 사용 가능 여부 확인
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # 모델 로드
#     model_path = './checkpoints/Face alignment with PAConv/custom/models/best_model.t7'
#     if not os.path.exists(model_path):
#         print(f"Error: Model file not found at {model_path}")
#         return
    
#     model = load_model(model_path, device)
#     print("Model loaded successfully!")
    
#     # 테스트 데이터셋 로드
#     test_dataset = FaceLandmarkData(data_dir='./dataset', partition='val')
#     print(f"Loaded {len(test_dataset)} test samples")
    
#     # 결과 저장 디렉토리 생성
#     os.makedirs('./results', exist_ok=True)
    
#     # 로그 파일 생성
#     log_file = './results/training_log.txt'
#     with open(log_file, 'w') as f:
#         f.write("Epoch\tTest Loss\tLandmark Error\n")
    
#     # 몇 개의 샘플에 대해 예측 수행
#     num_samples = min(5, len(test_dataset))  # 최대 5개 샘플
#     for i in range(num_samples):
#         points, true_landmarks, _ = test_dataset[i]
        
#         # 원본 파일 이름 가져오기
#         shape_file = test_dataset.shape_files[test_dataset.indices[i]]
#         landmark_file = test_dataset.landmark_files[test_dataset.indices[i]]
#         shape_name = os.path.basename(shape_file)
#         landmark_name = os.path.basename(landmark_file)
        
#         # 예측 수행
#         pred_landmarks = predict_landmarks(model, points, device)
        
#         # 결과 출력
#         print(f"\nSample {i+1}:")
#         print(f"Shape file: {shape_name}")
#         print(f"Landmark file: {landmark_name}")
#         print(f"Number of points: {len(points)}")
#         print(f"Number of landmarks: {len(pred_landmarks)}")
        
#         # 평균 오차 계산
#         error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks.numpy(), axis=1))
#         print(f"Average landmark error: {error:.4f}")
        
#         # 결과 저장
#         result_dict = {
#             'shape_file': shape_name,
#             'landmark_file': landmark_name,
#             'points': points.numpy(),
#             'true_landmarks': true_landmarks.numpy(),
#             'predicted_landmarks': pred_landmarks,
#             'error': error
#         }
        
#         # npy 파일로 저장
#         save_path = f'./results/sample_{i+1}_results.npy'
#         np.save(save_path, result_dict)
#         print(f"Saved results to {save_path}")
        
#         # 로그 파일에 기록
#         with open(log_file, 'a') as f:
#             f.write(f"{i+1}\t{error:.6f}\t{error:.6f}\n")

# if __name__ == "__main__":
#     # Training settings
#     args = parser.parse_args()
#     _init_(args)  # args를 전달
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#     train(args)
#     main()





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
from loss import AdaptiveWingLoss, HeatmapLoss
from util import main_sample
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate_fn(batch):
    points = torch.stack([item[0] for item in batch])
    landmarks = torch.stack([item[1] for item in batch])
    return points, landmarks, None

def generate_heatmap(points, landmarks, sigma=5.0):
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

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_landmark_error = 0
    
    with torch.no_grad():
        for data in test_loader:
            points, landmark = data
            points, landmark = points.to(device), landmark.to(device)
            
            # 모델 예측 (히트맵)
            points_normal = normalize_data(points)
            points_normal = points_normal.permute(0, 2, 1)
            pred_heatmap = model(points_normal)
            
            # 정답 히트맵 생성
            true_heatmap = generate_heatmap(points, landmark, sigma=3.0) # Use sigma=3 for eval as well
            
            # 히트맵 손실 계산
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            total_loss += heatmap_loss.item()

            # 예측 랜드마크 위치 추출
            pred_landmarks = get_predicted_landmarks_from_heatmap(points, pred_heatmap)

            # 랜드마크 오차 계산 (평균 거리)
            landmark_error = torch.norm(pred_landmarks - landmark, dim=2).mean()
            total_landmark_error += landmark_error.item()
    
    avg_loss = total_loss / len(test_loader)
    avg_landmark_error = total_landmark_error / len(test_loader)
    
    return avg_loss, avg_landmark_error

def train(args):
    _init_(args) # Initialize folders
    
    print(f"Number of points to sample: {args.num_points}")
    print(f"Number of landmarks: {args.num_landmarks}") # args.num_landmarks is 57

    # Convert absolute data_dir to relative if necessary
    if os.path.isabs(args.data_dir):
        args.data_dir = os.path.relpath(args.data_dir)

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
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)

    # Loss function (Heatmap loss)
    criterion = HeatmapLoss()

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
    
    for epoch in range(args.epoch):
        print(f'\nEpoch {epoch+1}/{args.epoch}')
        model.train()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader):
            points, landmark = data
            points, landmark = points.to(device), landmark.to(device)
            
            optimizer.zero_grad()

            # 모델 예측 (히트맵)
            points_normal = normalize_data(points) # Normalize points
            points_normal = points_normal.permute(0, 2, 1) # Adjust dimension for model (B, 3, N)
            pred_heatmap = model(points_normal)
            
            # 정답 히트맵 생성 (train에서는 sigma=3.0 사용)
            true_heatmap = generate_heatmap(points, landmark, sigma=3.0)
            
            # Heatmap 손실 계산
            heatmap_loss = criterion(pred_heatmap, true_heatmap)
            
            # 예측 랜드마크 위치 추출
            # Note: Use original points tensor (not normalized) for getting 3D coords
            pred_landmarks = get_predicted_landmarks_from_heatmap(points, pred_heatmap)

            # 랜드마크 위치 오차 계산 (Position Loss)
            # Use L2 norm (Euclidean distance) as position loss
            position_loss = torch.norm(pred_landmarks - landmark, dim=2).mean() # Average error per landmark per sample, then average over batch

            # 총 손실 계산 (Heatmap Loss + alpha * Position Loss)
            alpha = 0.1 # Weight for position loss
            total_loss = heatmap_loss + alpha * position_loss

            total_train_loss += total_loss.item()
            
            total_loss.backward()
            optimizer.step()
            
            # Print training loss periodically
            if (i + 1) % 10 == 0:
                print(f'  Iter {i+1}/{len(train_loader)}, Loss: {total_loss.item():.6f}')

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
    _init_(args)  # args를 전달
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train(args)
    main()




