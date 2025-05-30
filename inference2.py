import torch
import numpy as np
from My_args import *
from PAConv_model import PAConv
from dataset import FaceLandmarkData
from augmentations import normalize_data
import os

def load_model(model_path, device):
    # 모델 초기화
    args = parser.parse_args()
    model = PAConv(args, 57).to(device)  # 57 landmarks
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    
    state_dict = checkpoint
    # 만약 체크포인트가 dict형태면 'state_dict' 키를 확인
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    
    # module. 접두어가 있는지 확인
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # 'module.' 제거
        else:
            name = k
        new_state_dict[name] = v
    
    # 모델을 DataParallel로 감쌀지 여부에 따라 로드
    # 여기서는 DataParallel로 감싸고 로드
    model = torch.nn.DataParallel(model)
    model.load_state_dict(new_state_dict)
    
    model.eval()
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

if __name__ == "__main__":
    main()

# import torch
# import numpy as np
# from My_args import *
# from PAConv_model import PAConv
# from dataset import FaceLandmarkData
# from augmentations import normalize_data
# import os
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import torch.nn.functional as F

# def load_model(model_path, device):
#     # 모델 초기화
#     args = parser.parse_args()
#     model = PAConv(args, 57).to(device)  # 57 landmarks
    
#     # 학습된 가중치 로드
#     state_dict = torch.load(model_path)
#     # DataParallel로 저장된 모델의 경우 'module.' 접두사 제거
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if k.startswith('module.'):
#             new_state_dict[k[7:]] = v  # 'module.' 제거
#         else:
#             new_state_dict[k] = v
    
#     model.load_state_dict(new_state_dict)
#     model.eval()  # 평가 모드로 설정
#     return model

# def predict_landmarks(model, points, device):
#     with torch.no_grad():
#         points = points.unsqueeze(0).to(device)  # Add batch dimension
#         point_normal = normalize_data(points)
#         point_normal = point_normal.permute(0, 2, 1)
#         pred_heatmap = model(point_normal)
        
#         # Get landmark positions from heatmap
#         B, L, N = pred_heatmap.shape
#         pred_landmarks = torch.zeros(B, L, 3).to(device)
#         for b in range(B):
#             for l in range(L):
#                 max_idx = torch.argmax(pred_heatmap[b, l])
#                 pred_landmarks[b, l] = point_normal[b, :, max_idx]
        
#         return pred_landmarks[0].cpu().numpy(), pred_heatmap[0].cpu().numpy()

# def visualize_results(points, true_landmarks, pred_landmarks, save_path=None):
#     fig = plt.figure(figsize=(15, 5))
    
#     # 3D 뷰
#     ax1 = fig.add_subplot(121, projection='3d')
#     ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.1)
#     ax1.scatter(true_landmarks[:, 0], true_landmarks[:, 1], true_landmarks[:, 2], c='blue', s=50, label='True')
#     ax1.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], pred_landmarks[:, 2], c='red', s=50, label='Predicted')
#     ax1.set_title('3D View')
#     ax1.legend()
    
#     # 정면 뷰 (X-Y 평면)
#     ax2 = fig.add_subplot(122)
#     ax2.scatter(points[:, 0], points[:, 1], c='gray', s=1, alpha=0.1)
#     ax2.scatter(true_landmarks[:, 0], true_landmarks[:, 1], c='blue', s=50, label='True')
#     ax2.scatter(pred_landmarks[:, 0], pred_landmarks[:, 1], c='red', s=50, label='Predicted')
#     ax2.set_title('Front View (X-Y)')
#     ax2.legend()
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Saved visualization to {save_path}")
#     else:
#         plt.show()
    
#     plt.close()

# def visualize_heatmap(points, heatmap, save_path):
#     """히트맵을 시각화하고 저장하는 함수"""
#     # 3D 시각화
#     fig = plt.figure(figsize=(15, 5))
    
#     # 원본 포인트 클라우드
#     ax1 = fig.add_subplot(131, projection='3d')
#     ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.5)
#     ax1.set_title('Point Cloud')
    
#     # 히트맵 시각화 (첫 번째 랜드마크)
#     ax2 = fig.add_subplot(132, projection='3d')
#     scatter = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                          c=heatmap[0], cmap='hot', s=1)
#     plt.colorbar(scatter, ax=ax2)
#     ax2.set_title('Heatmap (Landmark 1)')
    
#     # 히트맵 시각화 (두 번째 랜드마크)
#     ax3 = fig.add_subplot(133, projection='3d')
#     scatter = ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                          c=heatmap[1], cmap='hot', s=1)
#     plt.colorbar(scatter, ax=ax3)
#     ax3.set_title('Heatmap (Landmark 2)')
    
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

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
#     os.makedirs('./results/heatmaps', exist_ok=True)
    
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
#         pred_landmarks, pred_heatmap = predict_landmarks(model, points, device)
        
#         # 결과 출력
#         print(f"\nSample {i+1}:")
#         print(f"Shape file: {shape_name}")
#         print(f"Landmark file: {landmark_name}")
#         print(f"Number of points: {len(points)}")
#         print(f"Number of landmarks: {len(pred_landmarks)}")
        
#         # 평균 오차 계산
#         error = np.mean(np.linalg.norm(pred_landmarks - true_landmarks.numpy(), axis=1))
#         print(f"Average landmark error: {error:.4f}")
        
#         # 히트맵 시각화 및 저장
#         heatmap_path = f'./results/heatmaps/sample_{i+1}_heatmap.png'
#         visualize_heatmap(points.numpy(), pred_heatmap, heatmap_path)
#         print(f"Saved heatmap visualization to {heatmap_path}")
        
#         # 결과 저장
#         result_dict = {
#             'shape_file': shape_name,
#             'landmark_file': landmark_name,
#             'points': points.numpy(),
#             'true_landmarks': true_landmarks.numpy(),
#             'predicted_landmarks': pred_landmarks,
#             'heatmap': pred_heatmap,
#             'error': error
#         }
        
#         # npy 파일로 저장
#         save_path = f'./results/sample_{i+1}_results.npy'
#         np.save(save_path, result_dict)
#         print(f"Saved results to {save_path}")

# if __name__ == "__main__":
#     main() 