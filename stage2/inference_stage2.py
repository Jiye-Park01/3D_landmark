import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse

# matplotlib 선택적 import
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization will be skipped.")

# Stage 1 모델 import
sys.path.append('/home/jhrew/jiye/3D_landmark')
from My_args import *
from PointTransformer_Stage2 import PointTransformerStage2
from dataset_stage2 import FaceLandmarkDataStage2, custom_collate_fn_stage2

def normalize_data(points):
    """포인트 클라우드 정규화 (중심 이동만, 스케일링 제거)"""
    centroid = torch.mean(points, dim=1, keepdim=True)
    points = points - centroid
    return points, centroid

def denormalize_landmarks(landmarks, centroid):
    """랜드마크를 원본 좌표계로 되돌리기 (중심 이동만 복원)"""
    landmarks = landmarks + centroid
    return landmarks

def load_model_stage2(model_path, args, landmark_num, device):
    """Stage 2 모델 로드"""
    model = PointTransformerStage2(args, landmark_num, pretrained_path=None)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
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

def predict_landmarks_stage2(model, points, device):
    """Stage 2 모델로 랜드마크 예측"""
    model.eval()
    with torch.no_grad():
        # 정규화 (원본 좌표계 정보 저장)
        points_normal, centroid = normalize_data(points)
        points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
        
        # 모델 예측
        heatmaps, coarse_landmarks, refined_landmarks = model(points_normal)
        
        # 예측 결과를 원본 좌표계로 되돌리기
        coarse_landmarks = denormalize_landmarks(coarse_landmarks, centroid)
        refined_landmarks = denormalize_landmarks(refined_landmarks, centroid)
        
        return heatmaps.cpu(), coarse_landmarks.cpu(), refined_landmarks.cpu()

def visualize_results_stage2(points, true_landmarks, pred_heatmaps, pred_coarse, pred_refined, save_path=None):
    """Stage 2 결과 시각화"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping visualization")
        return
    
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 포인트 클라우드 + 정답 랜드마크
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1], true_landmarks[:, 2], 
                c='red', s=50, marker='o', label='True Landmarks')
    ax1.set_title('Point Cloud + True Landmarks')
    ax1.legend()
    
    # 2. 포인트 클라우드 + Coarse 예측
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax2.scatter(pred_coarse[:, 0], pred_coarse[:, 1], pred_coarse[:, 2], 
                c='orange', s=50, marker='s', label='Coarse Prediction')
    ax2.set_title('Point Cloud + Coarse Prediction')
    ax2.legend()
    
    # 3. 포인트 클라우드 + Refined 예측
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax3.scatter(pred_refined[:, 0], pred_refined[:, 1], pred_refined[:, 2], 
                c='green', s=50, marker='^', label='Refined Prediction')
    ax3.scatter(true_landmarks[:, 0], true_landmarks[:, 1], true_landmarks[:, 2], 
                c='red', s=30, marker='o', alpha=0.7, label='True Landmarks')
    ax3.set_title('Point Cloud + Refined Prediction')
    ax3.legend()
    
    # 모든 subplot을 정면 뷰로 설정
    for ax in [ax1, ax2, ax3]:
        # 정면 뷰 설정 (얼굴이 정면을 향하도록)
        ax.view_init(elev=90, azim=0)  # 정면 (Y축에서 바라보기)
        
        # 축 비율 동일하게 설정
        try:
            ax.set_box_aspect([1,1,1])
        except:
            pass  # 오래된 matplotlib 버전 호환성
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_heatmap_stage2(points, heatmap, save_path=None):
    """Heatmap 시각화"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping heatmap visualization")
        return
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 포인트 클라우드 색상을 heatmap 값으로 설정
    colors = heatmap  # (N,) - 각 포인트의 heatmap 값
    
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, cmap='hot', s=2)
    
    plt.colorbar(scatter)
    ax.set_title('3D Heatmap Visualization')
    
    # 정면 뷰 설정
    ax.view_init(elev=90, azim=0)  # 정면 (Y축에서 바라보기)
    
    # 축 비율 동일하게 설정
    try:
        ax.set_box_aspect([1,1,1])
    except:
        pass  # 오래된 matplotlib 버전 호환성
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def calculate_metrics(pred_landmarks, true_landmarks):
    """평가 지표 계산"""
    # L2 거리 계산
    distances = np.linalg.norm(pred_landmarks - true_landmarks, axis=1)
    mean_error = np.mean(distances)
    
    # NME 계산 (얼굴 크기로 정규화)
    face_size = np.max(np.linalg.norm(true_landmarks, axis=1))
    nme = np.mean(distances / face_size)
    
    return mean_error, nme

def evaluate_full_dataset(model, test_dataset, device, batch_size=8):
    """전체 테스트 데이터셋에 대해 평가"""
    from torch.utils.data import DataLoader
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn_stage2,
        drop_last=False
    )
    
    model.eval()
    total_coarse_error = 0
    total_refined_error = 0
    total_coarse_nme = 0
    total_refined_nme = 0
    total_samples = 0
    
    print(f"Evaluating full dataset with {len(test_dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (points, landmarks, true_heatmaps) in enumerate(test_loader):
            points, landmarks = points.to(device), landmarks.to(device)
            batch_size_actual = points.size(0)
            
            # 정규화 (원본 좌표계 정보 저장)
            points_normal, centroid = normalize_data(points)
            points_normal = points_normal.permute(0, 2, 1)  # (B, 3, N)
            
            # 모델 예측
            pred_heatmaps, pred_coarse, pred_refined = model(points_normal)
            
            # 예측 결과를 원본 좌표계로 되돌리기
            pred_coarse = denormalize_landmarks(pred_coarse, centroid)
            pred_refined = denormalize_landmarks(pred_refined, centroid)
            
            # CPU로 변환
            landmarks = landmarks.cpu().numpy()
            pred_coarse = pred_coarse.cpu().numpy()
            pred_refined = pred_refined.cpu().numpy()
            
            # 배치 내 각 샘플에 대해 메트릭 계산
            for i in range(batch_size_actual):
                coarse_error, coarse_nme = calculate_metrics(pred_coarse[i], landmarks[i])
                refined_error, refined_nme = calculate_metrics(pred_refined[i], landmarks[i])
                
                total_coarse_error += coarse_error
                total_refined_error += refined_error
                total_coarse_nme += coarse_nme
                total_refined_nme += refined_nme
                total_samples += 1
            
            if batch_idx % 50 == 0:
                print(f"Processed {batch_idx * batch_size}/{len(test_dataset)} samples...")
    
    # 평균 계산
    avg_coarse_error = total_coarse_error / total_samples
    avg_refined_error = total_refined_error / total_samples
    avg_coarse_nme = total_coarse_nme / total_samples
    avg_refined_nme = total_refined_nme / total_samples
    
    return avg_coarse_error, avg_refined_error, avg_coarse_nme, avg_refined_nme, total_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/best_model_stage2.t7',
                       help='Stage 2 모델 경로')
    parser.add_argument('--data_dir', type=str, default='./dataset_stage2',
                       help='테스트 데이터 디렉토리')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='시각화할 샘플 수 (0이면 전체 데이터셋 평가만 수행)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--full_eval', action='store_true',
                       help='전체 테스트 데이터셋에 대해 평가 수행')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='전체 평가 시 배치 크기')
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
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
    
    # 모델 로드
    landmark_num = test_dataset.get_landmark_num()
    model = load_model_stage2(args.model_path, parser.parse_args([]), landmark_num, device)
    model = model.to(device)
    
    print(f"Loaded Stage 2 model with {landmark_num} landmarks")
    
    # 전체 데이터셋 평가 수행
    if args.full_eval or args.num_samples == 0:
        print("\n=== Full Dataset Evaluation ===")
        full_coarse_error, full_refined_error, full_coarse_nme, full_refined_nme, total_samples = evaluate_full_dataset(
            model, test_dataset, device, args.batch_size
        )
        
        print(f"\n=== Full Dataset Results ===")
        print(f"Total samples evaluated: {total_samples}")
        print(f"Average Coarse Error: {full_coarse_error:.4f}")
        print(f"Average Refined Error: {full_refined_error:.4f}")
        print(f"Average Coarse NME: {full_coarse_nme:.4f}")
        print(f"Average Refined NME: {full_refined_nme:.4f}")
        print(f"Improvement: {((full_coarse_error - full_refined_error) / full_coarse_error * 100):.2f}%")
        
        # 결과를 파일로 저장
        results_file = os.path.join(args.output_dir, 'full_dataset_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Stage 2 Full Dataset Evaluation Results\n")
            f.write(f"=====================================\n")
            f.write(f"Total samples evaluated: {total_samples}\n")
            f.write(f"Average Coarse Error: {full_coarse_error:.6f}\n")
            f.write(f"Average Refined Error: {full_refined_error:.6f}\n")
            f.write(f"Average Coarse NME: {full_coarse_nme:.6f}\n")
            f.write(f"Average Refined NME: {full_refined_nme:.6f}\n")
            f.write(f"Improvement: {((full_coarse_error - full_refined_error) / full_coarse_error * 100):.2f}%\n")
        
        print(f"Results saved to {results_file}")
        
        # 전체 평가만 수행하고 종료
        if args.num_samples == 0:
            return
    
    # 샘플별 시각화 수행
    if args.num_samples > 0:
        print(f"\n=== Sample Visualization (n={args.num_samples}) ===")
    
    # 테스트 실행
    total_coarse_error = 0
    total_refined_error = 0
    total_coarse_nme = 0
    total_refined_nme = 0
    
    # 특정 파일 5135F_FC_I 찾기
    selected_indices = []
    target_filename = "5135F_FC_I.npy"
    
    # 데이터셋에서 5135F_FC_I 파일 찾기
    found_target = False
    for i in range(len(test_dataset)):
        actual_idx = test_dataset.indices[i]
        pc_file, landmark_file = test_dataset.data_pairs[actual_idx]
        
        # 파일명 확인
        landmark_filename = os.path.basename(landmark_file)
        
        if landmark_filename == target_filename:
            selected_indices.append(i)
            found_target = True
            print(f"Found target file: {landmark_filename}")
            break
    
    if not found_target:
        print(f"Warning: Target file {target_filename} not found in test dataset!")
        print("Available files:")
        for i in range(min(10, len(test_dataset))):  # 처음 10개만 출력
            actual_idx = test_dataset.indices[i]
            pc_file, landmark_file = test_dataset.data_pairs[actual_idx]
            print(f"  {os.path.basename(landmark_file)}")
        print("  ...")
        
        # 타겟 파일을 찾지 못했으면 첫 번째 파일 사용
        selected_indices.append(0)
        print("Using first available file instead.")
    
    # 최종 선택된 표정과 사람 ID들
    final_expression_ids = set()
    final_person_ids = set()
    for i in selected_indices:
        actual_idx = test_dataset.indices[i]
        pc_file, landmark_file = test_dataset.data_pairs[actual_idx]
        filename = os.path.basename(landmark_file)
        person_id = filename.split('_')[0]
        expression_id = filename.split('_')[2].split('.')[0]  # 확장자 제거
        final_expression_ids.add(expression_id)
        final_person_ids.add(person_id)
    
    print(f"\nSelected {len(selected_indices)} samples:")
    print(f"  Expressions: {sorted(final_expression_ids)}")
    print(f"  Person IDs: {sorted(final_person_ids)}")
    
    for sample_idx, i in enumerate(selected_indices):
        print(f"\nProcessing sample {sample_idx + 1}/{len(selected_indices)}")
        
        # 데이터 로드 (선택된 인덱스 사용)
        points, landmarks, true_heatmaps = test_dataset[i]  # i는 이미 selected_indices에서 온 값
        
        # 파일 정보 출력 (데이터셋에서 실제 파일 경로 가져오기)
        actual_idx = test_dataset.indices[i]
        pc_file, landmark_file = test_dataset.data_pairs[actual_idx]
        
        # 파일명에서 정보 추출
        filename = os.path.basename(landmark_file)
        person_id = filename.split('_')[0]
        expression_id = filename.split('_')[2].split('.')[0]  # 확장자 제거
        
        print(f"  Person ID: {person_id}")
        print(f"  Expression: {expression_id}")
        print(f"  Point Cloud: {os.path.basename(pc_file)}")
        print(f"  Landmark: {os.path.basename(landmark_file)}")
        
        points = points.unsqueeze(0).to(device)  # (1, N, 3)
        landmarks = landmarks.unsqueeze(0).to(device)  # (1, L, 3)
        
        # 예측
        pred_heatmaps, pred_coarse, pred_refined = predict_landmarks_stage2(model, points, device)
        
        # CPU로 변환
        points = points.cpu().squeeze(0).numpy()  # (N, 3)
        landmarks = landmarks.cpu().squeeze(0).numpy()  # (L, 3)
        pred_coarse = pred_coarse.squeeze(0).numpy()  # (L, 3)
        pred_refined = pred_refined.squeeze(0).numpy()  # (L, 3)
        
        # 원본 좌표계로 역정규화 (중심 이동만 복원)
        # 원본 포인트 클라우드 로드
        original_points = np.load(pc_file)
        
        # 정규화 파라미터 계산 (데이터셋과 동일한 방식)
        centroid = np.mean(original_points, axis=0)
        
        # 역정규화 함수 (중심 이동만 복원)
        def denormalize(normalized_data):
            return normalized_data + centroid
        
        # 모든 데이터를 원본 좌표계로 역정규화
        points = denormalize(points)
        landmarks = denormalize(landmarks)
        pred_coarse = denormalize(pred_coarse)
        pred_refined = denormalize(pred_refined)
        
        # 평가 지표 계산
        coarse_error, coarse_nme = calculate_metrics(pred_coarse, landmarks)
        refined_error, refined_nme = calculate_metrics(pred_refined, landmarks)
        
        total_coarse_error += coarse_error
        total_refined_error += refined_error
        total_coarse_nme += coarse_nme
        total_refined_nme += refined_nme
        
        print(f"Sample {sample_idx + 1} ({person_id}_{expression_id}) - Coarse Error: {coarse_error:.4f}, Refined Error: {refined_error:.4f}")
        print(f"Sample {sample_idx + 1} ({person_id}_{expression_id}) - Coarse NME: {coarse_nme:.4f}, Refined NME: {refined_nme:.4f}")
        
        # 시각화 저장
        # 1. Heatmap 시각화 (첫 번째 랜드마크)
        heatmap_path = os.path.join(args.output_dir, 'heatmaps', f'sample_{sample_idx + 1}_{person_id}_{expression_id}_heatmap.png')
        visualize_heatmap_stage2(points, pred_heatmaps[0, 0].numpy(), heatmap_path)
        
        # 2. 3D 결과 시각화
        result_path = os.path.join(args.output_dir, 'visualizations', f'sample_{sample_idx + 1}_{person_id}_{expression_id}_result.png')
        visualize_results_stage2(points, landmarks, pred_heatmaps[0], 
                               pred_coarse, pred_refined, result_path)
        
        # 결과 저장 (.npy)
        result_data = {
            'points': points,
            'true_landmarks': landmarks,
            'pred_coarse': pred_coarse,
            'pred_refined': pred_refined,
            'pred_heatmaps': pred_heatmaps.numpy(),
            'coarse_error': coarse_error,
            'refined_error': refined_error,
            'coarse_nme': coarse_nme,
            'refined_nme': refined_nme,
            'pc_file': pc_file,
            'landmark_file': landmark_file,
            'person_id': person_id,
            'expression_id': expression_id
        }
        np.save(os.path.join(args.output_dir, f'sample_{sample_idx + 1}_{person_id}_{expression_id}_results.npy'), result_data)
    
    # 평균 결과 출력
    avg_coarse_error = total_coarse_error / len(selected_indices)
    avg_refined_error = total_refined_error / len(selected_indices)
    avg_coarse_nme = total_coarse_nme / len(selected_indices)
    avg_refined_nme = total_refined_nme / len(selected_indices)
    
    print(f"\n=== Stage 2 Evaluation Results ===")
    print(f"Tested expressions: {sorted(final_expression_ids)}")
    print(f"Average Coarse Error: {avg_coarse_error:.4f}")
    print(f"Average Refined Error: {avg_refined_error:.4f}")
    print(f"Average Coarse NME: {avg_coarse_nme:.4f}")
    print(f"Average Refined NME: {avg_refined_nme:.4f}")
    print(f"Improvement: {((avg_coarse_error - avg_refined_error) / avg_coarse_error * 100):.2f}%")

if __name__ == '__main__':
    main() 