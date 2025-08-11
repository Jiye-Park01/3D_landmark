import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import argparse

def visualize_pointcloud_with_landmarks(points, landmarks, save_path=None, title="Point Cloud with Landmarks"):
    """포인트 클라우드와 랜드마크를 3D로 시각화"""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 포인트 클라우드 시각화 (작은 점들)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c='lightblue', s=1, alpha=0.6, label='Point Cloud')
    
    # 랜드마크 시각화 (큰 빨간 점들)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], 
               c='red', s=50, marker='o', label='Landmarks')
    
    # 랜드마크 번호 표시 (선택적으로)
    for i, landmark in enumerate(landmarks):
        if i % 10 == 0:  # 10개마다 번호 표시 (너무 많으면 복잡해짐)
            ax.text(landmark[0], landmark[1], landmark[2], 
                   f'{i}', fontsize=8, color='darkred')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # 축 범위 설정 (포인트 클라우드와 랜드마크를 모두 포함하도록)
    x_min, x_max = min(points[:, 0].min(), landmarks[:, 0].min()), max(points[:, 0].max(), landmarks[:, 0].max())
    y_min, y_max = min(points[:, 1].min(), landmarks[:, 1].min()), max(points[:, 1].max(), landmarks[:, 1].max())
    z_min, z_max = min(points[:, 2].min(), landmarks[:, 2].min()), max(points[:, 2].max(), landmarks[:, 2].max())
    
    # 약간의 여백 추가
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_multiple_views(points, landmarks, save_dir, filename):
    """여러 각도에서 시각화"""
    
    # 1. 정면도 (X-Y 평면)
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='red', s=50, marker='o')
    ax1.set_title('Front View (X-Y)')
    ax1.view_init(elev=0, azim=0)
    
    # 2. 측면도 (Y-Z 평면)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='red', s=50, marker='o')
    ax2.set_title('Side View (Y-Z)')
    ax2.view_init(elev=0, azim=90)
    
    # 3. 상면도 (X-Z 평면)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(points[:, 0], points[:, 1], points[:, 2], c='lightblue', s=1, alpha=0.6)
    ax3.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='red', s=50, marker='o')
    ax3.set_title('Top View (X-Z)')
    ax3.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{filename}_multi_view.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved multi-view visualization to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.', help='데이터 디렉토리')
    parser.add_argument('--landmarks_dir', type=str, default='./landmarks', help='랜드마크 디렉토리')
    parser.add_argument('--sample_idx', type=int, default=0, help='시각화할 샘플 인덱스')
    parser.add_argument('--output_dir', type=str, default='.', help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 데이터 파일들 로드
    landmark_files = sorted(glob.glob(os.path.join(args.landmarks_dir, '*.npy')))
    shapes_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'shapes', '*')))
    
    if args.sample_idx >= len(landmark_files):
        print(f"Error: sample_idx {args.sample_idx} is out of range. Total samples: {len(landmark_files)}")
        return
    
    # 선택된 샘플 로드
    landmark_file = landmark_files[args.sample_idx]
    shape_dir = shapes_dirs[args.sample_idx]
    
    # 파일명에서 ID 추출
    filename = os.path.basename(landmark_file)
    person_id = filename.split('_')[0]
    expression_id = filename.split('_')[2]
    
    print(f"Visualizing sample {args.sample_idx + 1}: {filename}")
    print(f"Person ID: {person_id}, Expression: {expression_id}")
    
    # 랜드마크 로드
    landmarks = np.load(landmark_file)
    print(f"Landmarks shape: {landmarks.shape}")
    
    # 포인트 클라우드 파일 찾기
    pc_files = glob.glob(os.path.join(shape_dir, f'{person_id}_FC_{expression_id}_pc.npy'))
    
    if not pc_files:
        print(f"Warning: No matching PC file found for {person_id}_FC_{expression_id}")
        # 첫 번째 PC 파일 사용
        pc_files = glob.glob(os.path.join(shape_dir, '*.npy'))
        if pc_files:
            print(f"Using first available PC file: {os.path.basename(pc_files[0])}")
        else:
            print("Error: No PC files found in shape directory")
            return
    
    # 포인트 클라우드 로드
    points = np.load(pc_files[0])
    print(f"Points shape: {points.shape}")
    
    # 좌표계 정보 출력
    print(f"\nCoordinate System Info:")
    print(f"Landmarks - X: [{landmarks[:, 0].min():.3f}, {landmarks[:, 0].max():.3f}], "
          f"Y: [{landmarks[:, 1].min():.3f}, {landmarks[:, 1].max():.3f}], "
          f"Z: [{landmarks[:, 2].min():.3f}, {landmarks[:, 2].max():.3f}]")
    print(f"Points - X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
          f"Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
          f"Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # 중심점 비교
    landmark_center = landmarks.mean(axis=0)
    points_center = points.mean(axis=0)
    print(f"Landmark center: [{landmark_center[0]:.3f}, {landmark_center[1]:.3f}, {landmark_center[2]:.3f}]")
    print(f"Points center: [{points_center[0]:.3f}, {points_center[1]:.3f}, {points_center[2]:.3f}]")
    
    # 시각화
    base_filename = f"sample_{args.sample_idx + 1}_{person_id}_{expression_id}"
    
    # 1. 기본 시각화
    save_path = os.path.join(args.output_dir, f'{base_filename}_basic.png')
    visualize_pointcloud_with_landmarks(
        points, landmarks, save_path, 
        f"Point Cloud with Landmarks - {person_id}_{expression_id}"
    )
    
    # 2. 다중 각도 시각화
    visualize_multiple_views(points, landmarks, args.output_dir, base_filename)
    
    print(f"\nVisualization completed!")
    print(f"Files saved in: {args.output_dir}")

if __name__ == '__main__':
    main()