import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.append('/home/jhrew/jiye/3D_pointtransformer')

# Stage 1 모델 import
sys.path.append('/home/jhrew/jiye/3D_landmark')
from My_args import *
from PointTransformer_Stage2 import PointTransformerStage2
from dataset_stage2 import FaceLandmarkDataStage2, custom_collate_fn_stage2

def normalize_data(points):
    """포인트 클라우드 정규화 (중심 이동만, 스케일링 제거)"""
    centroid = torch.mean(points, dim=1, keepdim=True)
    points = points - centroid
    return points

def calculate_landmark_error(pred_landmarks, true_landmarks):
    """랜드마크 오차 계산 (Chamfer Distance 기반)"""
    # 각 랜드마크 간의 L2 거리 계산
    distances = torch.norm(pred_landmarks - true_landmarks, dim=2)  # (B, L)
    mean_error = torch.mean(distances)
    return mean_error.item()

def calculate_nme(pred_landmarks, true_landmarks, face_size):
    """Normalized Mean Error 계산"""
    distances = torch.norm(pred_landmarks - true_landmarks, dim=2)  # (B, L)
    nme = torch.mean(distances / face_size)
    return nme.item()

def calculate_registration_loss(pred_landmarks, true_landmarks):
    """Registration Loss 계산 - 예측 랜드마크와 정답 랜드마크 간의 정렬 오차"""
    # 각 랜드마크 간의 L2 거리 계산
    distances = torch.norm(pred_landmarks - true_landmarks, dim=2)  # (B, L)
    registration_loss = torch.mean(distances)
    return registration_loss.item()

def evaluate_stage2(model, test_loader, device, args):
    """Stage 2 모델 평가"""
    model.eval()
    total_heatmap_loss = 0
    total_regression_loss = 0
    total_unfolding_loss = 0
    total_registration_loss = 0
    total_landmark_error = 0
    total_nme = 0
    num_batches = 0
    
    heatmap_criterion = nn.MSELoss()
    regression_criterion = nn.MSELoss()
    
    with torch.no_grad():
        for points, landmarks, true_heatmaps in test_loader:
            points, landmarks, true_heatmaps = points.to(device), landmarks.to(device), true_heatmaps.to(device)
            
            # 정규화
            points_normal = normalize_data(points)
            points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 모델 예측
            pred_heatmaps, coarse_landmarks, refined_landmarks = model(points_normal)
            
            # 1. Heatmap Loss
            heatmap_loss = heatmap_criterion(pred_heatmaps, true_heatmaps)
            
            # 2. Regression Loss (refined landmarks 사용)
            regression_loss = regression_criterion(refined_landmarks, landmarks)
            
            # 3. Unfolding Loss (조건부)
            unfolding_loss = 0
            if args.use_unfolding_loss:
                unfolding_loss = torch.mean(torch.norm(refined_landmarks - coarse_landmarks, dim=2))
            
            # 4. Registration Loss
            registration_loss = calculate_registration_loss(refined_landmarks, landmarks)
            
            # 5. Landmark Error 계산
            landmark_error = calculate_landmark_error(refined_landmarks, landmarks)
            
            # 6. NME 계산 (얼굴 크기로 정규화)
            face_size = torch.max(torch.norm(landmarks, dim=2))  # 얼굴 크기 추정
            nme = calculate_nme(refined_landmarks, landmarks, face_size)
            
            total_heatmap_loss += heatmap_loss.item()
            total_regression_loss += regression_loss.item()
            total_unfolding_loss += unfolding_loss.item() if args.use_unfolding_loss else 0
            total_registration_loss += registration_loss
            total_landmark_error += landmark_error
            total_nme += nme
            num_batches += 1
    
    avg_heatmap_loss = total_heatmap_loss / num_batches
    avg_regression_loss = total_regression_loss / num_batches
    avg_unfolding_loss = total_unfolding_loss / num_batches if args.use_unfolding_loss else 0
    avg_registration_loss = total_registration_loss / num_batches
    avg_landmark_error = total_landmark_error / num_batches
    avg_nme = total_nme / num_batches
    
    return avg_heatmap_loss, avg_regression_loss, avg_unfolding_loss, avg_registration_loss, avg_landmark_error, avg_nme

def train_stage2(args):
    """Stage 2 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 데이터셋 로드
    train_dataset = FaceLandmarkDataStage2(
        data_dir=args.data_dir,
        sigma=args.sigma,
        num_points=args.num_points,
        split='train'
    )
    
    test_dataset = FaceLandmarkDataStage2(
        data_dir=args.data_dir,
        sigma=args.sigma,
        num_points=args.num_points,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        collate_fn=custom_collate_fn_stage2,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        collate_fn=custom_collate_fn_stage2,
        drop_last=True
    )
    
    # 모델 생성 (Stage 1 모델 로드)
    landmark_num = train_dataset.get_landmark_num()
    model = PointTransformerStage2(args, landmark_num, pretrained_path=args.stage1_model_path)
    model = model.to(device)
    
    # Multi-GPU 지원
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Loss 함수들
    heatmap_criterion = nn.MSELoss()
    regression_criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Early Stopping (Regression Loss 기준)
    patience = 20
    min_delta = 1e-4
    patience_counter = 0
    best_regression_loss = float('inf')
    
    # 로그 파일 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = f'./checkpoints/training_log_{timestamp}.txt'
    training_log_filepath = f'./checkpoints/training_console_log_{timestamp}.txt'
    
    # 체크포인트 저장 경로
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 로그 헤더 작성
    with open(log_filepath, 'w') as f:
        f.write('Epoch\tHeatmap Loss\tRegression Loss\tUnfolding Loss\tRegistration Loss\tTotal Loss\tLandmark Error\tNME\n')
    
    # Training console log 파일 초기화
    with open(training_log_filepath, 'w') as f:
        f.write(f"Stage 2 Training Log - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Loss configuration:\n")
        f.write(f"  - Heatmap Loss: 0.5\n")
        f.write(f"  - Regression Loss: 1.5\n")
        f.write(f"  - Unfolding Loss: {0.1 if args.use_unfolding_loss else 0}\n")
        f.write(f"  - Registration Loss: {0.1 if args.use_registration_loss else 0}\n")
        f.write(f"Early Stopping: Regression Loss 기준 (patience=20)\n")
        f.write("="*50 + "\n\n")
    
    print("Start Stage 2 training...")
    print(f"Loss configuration:")
    print(f"  - Heatmap Loss: 0.5")
    print(f"  - Regression Loss: 1.5")
    print(f"  - Unfolding Loss: {0.1 if args.use_unfolding_loss else 0}")
    print(f"  - Registration Loss: {0.1 if args.use_registration_loss else 0}")
    
    for epoch in range(args.epochs):
        epoch_start_time = datetime.now()
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Training console log에 에포크 시작 기록
        with open(training_log_filepath, 'a') as f:
            f.write(f"\nEpoch {epoch+1}/{args.epochs} - {epoch_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 30 + "\n")
        
        # 학습
        model.train()
        total_heatmap_loss = 0
        total_regression_loss = 0
        total_unfolding_loss = 0
        total_registration_loss = 0
        total_landmark_error = 0
        num_batches = 0
        
        for batch_idx, (points, landmarks, true_heatmaps) in enumerate(train_loader):
            points, landmarks, true_heatmaps = points.to(device), landmarks.to(device), true_heatmaps.to(device)
            
            # 정규화
            points_normal = normalize_data(points)
            points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 모델 예측
            pred_heatmaps, coarse_landmarks, refined_landmarks = model(points_normal)
            
            # Loss 계산
            heatmap_loss = heatmap_criterion(pred_heatmaps, true_heatmaps)
            regression_loss = regression_criterion(refined_landmarks, landmarks)
            
            # 추가 Loss 계산 (조건부)
            unfolding_loss = 0
            registration_loss = 0
            
            if args.use_unfolding_loss:
                # Unfolding Loss 계산 (간단한 구현)
                unfolding_loss = torch.mean(torch.norm(refined_landmarks - coarse_landmarks, dim=2))
            
            if args.use_registration_loss:
                # Registration Loss 계산
                registration_loss = torch.mean(torch.norm(refined_landmarks - landmarks, dim=2))
            
            # 랜드마크 에러 계산 (학습 중 모니터링용)
            landmark_error = calculate_landmark_error(refined_landmarks, landmarks)
            
            # 가중치 적용된 Total Loss
            total_loss = (
                0.5 * heatmap_loss +
                1.5 * regression_loss +
                (0.1 * unfolding_loss if args.use_unfolding_loss else 0) +
                (0.1 * registration_loss if args.use_registration_loss else 0)
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_heatmap_loss += heatmap_loss.item()
            total_regression_loss += regression_loss.item()
            total_unfolding_loss += unfolding_loss.item() if args.use_unfolding_loss else 0
            total_registration_loss += registration_loss.item() if args.use_registration_loss else 0
            total_landmark_error += landmark_error
            num_batches += 1
            
            if batch_idx % 10 == 0:
                batch_log = f'Batch {batch_idx}/{len(train_loader)}, Heatmap Loss: {heatmap_loss.item():.4f}, Regression Loss: {regression_loss.item():.4f}, Landmark Error: {landmark_error:.4f}, Total Loss: {total_loss.item():.4f}'
                print(batch_log)
                # Training console log에 배치 로그 저장
                with open(training_log_filepath, 'a') as f:
                    f.write(f"  {batch_log}\n")
        
        # 평균 Loss 및 Landmark Error 계산
        avg_heatmap_loss = total_heatmap_loss / num_batches
        avg_regression_loss = total_regression_loss / num_batches
        avg_unfolding_loss = total_unfolding_loss / num_batches if args.use_unfolding_loss else 0
        avg_registration_loss = total_registration_loss / num_batches if args.use_registration_loss else 0
        avg_landmark_error = total_landmark_error / num_batches
        avg_total_loss = (
            0.5 * avg_heatmap_loss +
            1.5 * avg_regression_loss +
            (0.1 * avg_unfolding_loss if args.use_unfolding_loss else 0) +
            (0.1 * avg_registration_loss if args.use_registration_loss else 0)
        )
        
        # 평가
        test_heatmap_loss, test_regression_loss, test_unfolding_loss, test_registration_loss, test_landmark_error, test_nme = evaluate_stage2(
            model, test_loader, device, args
        )
        test_total_loss = (
            0.5 * test_heatmap_loss +
            1.5 * test_regression_loss +
            (0.1 * test_unfolding_loss if args.use_unfolding_loss else 0) +
            (0.1 * test_registration_loss if args.use_registration_loss else 0)
        )
        
        # 로그 기록
        with open(log_filepath, 'a') as f:
            f.write(f'{epoch+1}\t{test_heatmap_loss:.6f}\t{test_regression_loss:.6f}\t{test_unfolding_loss:.6f}\t{test_registration_loss:.6f}\t'
                   f'{test_total_loss:.6f}\t{test_landmark_error:.6f}\t{test_nme:.6f}\n')
        
        # Train 및 Test 결과 출력 및 저장
        train_log = f'Train - Heatmap Loss: {avg_heatmap_loss:.4f}, Regression Loss: {avg_regression_loss:.4f}, Landmark Error: {avg_landmark_error:.4f}, Total Loss: {avg_total_loss:.4f}'
        test_log = f'Test - Heatmap Loss: {test_heatmap_loss:.4f}, Regression Loss: {test_regression_loss:.4f}, Unfolding Loss: {test_unfolding_loss:.4f}, Registration Loss: {test_registration_loss:.4f}, Total Loss: {test_total_loss:.4f}, Landmark Error: {test_landmark_error:.4f}, NME: {test_nme:.4f}'
        print(train_log)
        print(test_log)
        
        # Training console log에 결과 저장
        with open(training_log_filepath, 'a') as f:
            f.write(f"  {train_log}\n")
            f.write(f"  {test_log}\n")
        
        # 10번 에포크마다 모델 저장
        if (epoch + 1) % 10 == 0:
            epoch_model_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_model_stage2.t7')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'test_regression_loss': test_regression_loss,
                'test_total_loss': test_total_loss,
                'args': args
            }, epoch_model_path)
            
            epoch_save_log = f"Saved model at epoch {epoch+1} to {epoch_model_path}"
            print(epoch_save_log)
            
            # Training console log에 에포크 모델 저장 기록
            with open(training_log_filepath, 'a') as f:
                f.write(f"  {epoch_save_log}\n")
        
        # Early Stopping (Regression Loss 기준)
        if test_regression_loss < best_regression_loss - min_delta:
            best_regression_loss = test_regression_loss
            patience_counter = 0
            
            # Best model 저장
            best_model_path = os.path.join(checkpoint_dir, 'best_model_stage2.t7')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_regression_loss': best_regression_loss,
                'test_regression_loss': test_regression_loss,
                'test_total_loss': test_total_loss,
                'args': args
            }, best_model_path)
            
            best_log = f"Saved best model to {best_model_path} (Regression Loss: {best_regression_loss:.6f})"
            print(best_log)
            
            # Training console log에 best model 저장 기록
            with open(training_log_filepath, 'a') as f:
                f.write(f"  {best_log}\n")
        else:
            patience_counter += 1
            patience_log = f"Patience counter: {patience_counter}/{patience}"
            print(patience_log)
            
            # Training console log에 patience 기록
            with open(training_log_filepath, 'a') as f:
                f.write(f"  {patience_log}\n")
        
        # Early stopping 체크
        if patience_counter >= patience:
            early_stop_log = f"Early stopping triggered after {epoch+1} epochs!"
            print(early_stop_log)
            
            # Training console log에 early stopping 기록
            with open(training_log_filepath, 'a') as f:
                f.write(f"\n{early_stop_log}\n")
            break
        
        # Scheduler step
        scheduler.step()
        
        # 에포크 완료 시간 기록
        epoch_end_time = datetime.now()
        epoch_duration = epoch_end_time - epoch_start_time
        
        with open(training_log_filepath, 'a') as f:
            f.write(f"  Epoch {epoch+1} completed in {epoch_duration}\n")
            if (epoch + 1) % 10 == 0:
                f.write(f"  Model saved: {epoch_model_path}\n")
    
    # 학습 완료 로그
    completion_log = "Stage 2 training finished!"
    print(completion_log)
    
    with open(training_log_filepath, 'a') as f:
        f.write(f"\n{completion_log}\n")
        f.write(f"Final best regression loss: {best_regression_loss:.6f}\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 데이터 관련
    parser.add_argument('--data_dir', type=str, default='./dataset_stage2', 
                       help='Stage 2 데이터 디렉토리')
    parser.add_argument('--stage1_model_path', type=str, 
                       default='../checkpoints/PointTransformer_Landmark_Detection/custom/models/best_model.t7',
                       help='Stage 1 모델 경로')
    
    # 모델 관련
    parser.add_argument('--num_points', type=int, default=8192, help='포인트 수')
    parser.add_argument('--sigma', type=float, default=1.0, help='Heatmap sigma')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--lr', type=float, default=1e-3, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--workers', type=int, default=4, help='데이터 로더 워커 수')
    parser.add_argument('--use_unfolding_loss', action='store_true', help='Unfolding Loss 사용')
    parser.add_argument('--use_registration_loss', action='store_true', help='Registration Loss 사용')
    
    args = parser.parse_args()
    
    # 학습 실행
    model = train_stage2(args) 