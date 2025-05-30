import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import FaceLandmarkData
from PAConv_model import PAConv
from My_args import parse_args

def visualize_prediction(points, true_landmarks, pred_landmarks, epoch, save_dir='./visualizations'):
    """
    포인트 클라우드와 랜드마크를 시각화하는 함수
    z축 기준으로 위에서 아래로 내려다보는 형태로 시각화
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 포인트 클라우드 시각화 (회색으로)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c='gray', alpha=0.1, s=1, label='Point Cloud')
    
    # 실제 랜드마크 시각화 (빨간색)
    ax.scatter(true_landmarks[:, 0], true_landmarks[:, 1], true_landmarks[:, 2],
              c='red', s=50, label='True Landmarks')
    
    # 예측 랜드마크 시각화 (파란색)
    ax.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], pred_landmarks[:, 2],
              c='blue', s=50, label='Predicted Landmarks')
    
    # 뷰 설정 (z축 기준 위에서 아래로)
    ax.view_init(elev=90, azim=0)
    
    # 축 레이블 설정
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 제목 설정
    plt.title(f'Landmark Prediction (Epoch {epoch})')
    plt.legend()
    
    # 저장
    plt.savefig(os.path.join(save_dir, f'prediction_epoch_{epoch}.png'))
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 모델 로드
    model_path = './checkpoints/Face alignment with PAConv/custom/models/best_model.t7'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = PAConv(args, args.num_landmarks).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 테스트 데이터셋 로드
    test_dataset = FaceLandmarkData(data_dir='./dataset', partition='val')
    
    # 몇 개의 샘플에 대해 시각화
    num_samples = min(5, len(test_dataset))
    for i in range(num_samples):
        points, true_landmarks, _ = test_dataset[i]
        
        # 모델 입력 형태로 변환
        points = points.unsqueeze(0).permute(0, 2, 1).to(device)  # (1, 3, N)
        
        # 예측
        with torch.no_grad():
            pred_heatmap = model(points)
            pred_landmarks = get_predicted_landmarks_from_heatmap(points.permute(0, 2, 1), pred_heatmap)
        
        # CPU로 이동하고 numpy로 변환
        points_np = points[0].permute(1, 0).cpu().numpy()  # (N, 3)
        true_landmarks_np = true_landmarks.numpy()  # (L, 3)
        pred_landmarks_np = pred_landmarks[0].cpu().numpy()  # (L, 3)
        
        # 시각화
        visualize_prediction(points_np, true_landmarks_np, pred_landmarks_np, 
                           epoch=i, save_dir='./visualizations')

if __name__ == "__main__":
    main() 