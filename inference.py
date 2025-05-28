import torch
import numpy as np
from My_args import *
from PAConv_model import PAConv
from dataset import FaceLandmarkData
from augmentations import normalize_data
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 새로 만든거(실행)
# matplotlib으로 간단한 시각화

def load_model(model_path, device):
    # 모델 초기화
    args = parser.parse_args()
    model = PAConv(args, 57).to(device)  # 57 landmarks
    model = torch.nn.DataParallel(model)
    
    # 학습된 가중치 로드
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    return model

def predict_landmarks(model, point_cloud, device):
    # 데이터 전처리
    point_cloud = torch.FloatTensor(point_cloud).unsqueeze(0)  # 배치 차원 추가
    point_cloud = point_cloud.to(device)
    
    # 정규화
    point_normal = normalize_data(point_cloud)
    point_normal = point_normal.permute(0, 2, 1)  # (1, 3, N)
    
    # 추론
    with torch.no_grad():
        pred_heatmap = model(point_normal)
    
    # 히트맵에서 랜드마크 위치 추출
    B, L, N = pred_heatmap.shape
    pred_landmarks = torch.zeros(B, L, 3).to(device)
    for b in range(B):
        for l in range(L):
            max_idx = torch.argmax(pred_heatmap[b, l])
            pred_landmarks[b, l] = point_normal[b, :, max_idx]
    
    return pred_landmarks[0].cpu().numpy()  # 배치 차원 제거하고 numpy로 변환

def visualize_results(points, true_landmarks, pred_landmarks, save_path=None):
    fig = plt.figure(figsize=(15, 5))
    
    # 3D 뷰
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.1)
    ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1], true_landmarks[:, 2], c='blue', s=50, label='True')
    ax1.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], pred_landmarks[:, 2], c='red', s=50, label='Predicted')
    ax1.set_title('3D View')
    ax1.legend()
    
    # 정면 뷰 (X-Y 평면)
    ax2 = fig.add_subplot(122)
    ax2.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.1)
    ax2.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='blue', s=50, label='True')
    ax2.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=50, label='Predicted')
    ax2.set_title('Front View (X-Y)')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

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
    
    # 몇 개의 샘플에 대해 예측 수행
    num_samples = min(5, len(test_dataset))  # 최대 5개 샘플
    for i in range(num_samples):
        points, true_landmarks, _ = test_dataset[i]
        
        # 예측 수행
        pred_landmarks = predict_landmarks(model, points, device)
        
        # 결과 출력
        print(f"\nSample {i+1}:")
        print(f"Number of points: {len(points)}")
        print(f"Number of landmarks: {len(pred_landmarks)}")
        
        # 평균 오차 계산
        error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks.numpy(), axis=1))
        print(f"Average landmark error: {error:.4f}")
        
        # 결과 시각화
        save_path = f'./results/sample_{i+1}.png'
        visualize_results(points.numpy(), true_landmarks.numpy(), pred_landmarks, save_path)

if __name__ == "__main__":
    main() 