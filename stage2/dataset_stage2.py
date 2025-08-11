import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist

class FaceLandmarkDataStage2(Dataset):
    """Stage 2: 다양한 표정 기반 3D 랜드마크 데이터셋"""
    
    def __init__(self, data_dir, sigma=1.0, num_points=8192, split='train'):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            sigma: heatmap 생성 시 사용할 Gaussian sigma 값
            num_points: 샘플링할 포인트 수
            split: 'train' 또는 'test'
        """
        self.data_dir = data_dir
        self.sigma = sigma
        self.num_points = num_points
        self.split = split
        
        # landmarks 폴더에서 랜드마크 파일들 로드
        landmark_files = sorted(glob.glob(os.path.join(data_dir, 'landmarks', '*.npy')))
        
        # 각 랜드마크 파일에 대응하는 포인트 클라우드 파일들 찾기
        self.data_pairs = []
        
        for landmark_file in landmark_files:
            # 파일명에서 사람 ID 추출 (예: 1001M_FC_A.npy -> 1001M)
            filename = os.path.basename(landmark_file)
            person_id = filename.split('_')[0]  # 1001M
            expression_id = filename.split('_')[2].split('.')[0]  # A (확장자 제거)
            
            # 해당 사람의 shapes 폴더에서 모든 포인트 클라우드 파일 찾기
            person_shapes_dir = os.path.join(data_dir, 'shapes', person_id)
            if os.path.exists(person_shapes_dir):
                pc_files = glob.glob(os.path.join(person_shapes_dir, f'{person_id}_FC_*_pc.npy'))
                
                for pc_file in pc_files:
                    # 포인트 클라우드 파일명에서 표정 ID 추출
                    pc_filename = os.path.basename(pc_file)
                    pc_expression_id = pc_filename.split('_')[2]  # A, B, C, ...
                    
                    # 랜드마크와 포인트 클라우드 매칭
                    if pc_expression_id == expression_id:
                        self.data_pairs.append((pc_file, landmark_file))
                        break  # 첫 번째 매칭만 사용
        
        # Train/Test 분할 (80:20)
        total_samples = len(self.data_pairs)
        train_end_idx = int(0.8 * total_samples)
        
        if split == 'train':
            self.indices = list(range(0, train_end_idx))
        else:  # test
            self.indices = list(range(train_end_idx, total_samples))
        
        print(f"Stage 2 {split} dataset: {len(self.indices)} samples")
        print(f"Total data pairs found: {len(self.data_pairs)}")
        
        # 테스트 데이터셋 파일 목록 저장 (비교실험용)
        if split == 'test':
            self.save_test_dataset_info()
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 실제 인덱스 계산
        actual_idx = self.indices[idx]
        pc_file, landmark_file = self.data_pairs[actual_idx]
        
        # 포인트 클라우드 로드
        points = np.load(pc_file)  # (N, 3)
        
        # 랜드마크 로드
        landmarks = np.load(landmark_file)  # (L, 3)
        
        # 포인트 수 조정
        if len(points) > self.num_points:
            # 랜덤 샘플링
            indices = np.random.choice(len(points), self.num_points, replace=False)
            points = points[indices]
        elif len(points) < self.num_points:
            # 중복 샘플링으로 보충
            indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
        
        # 포인트 클라우드와 랜드마크를 함께 정규화
        points, landmarks = self.normalize_points_and_landmarks(points, landmarks)
        
        # Heatmap 생성
        heatmaps = self.generate_heatmaps(points, landmarks)
        
        return torch.FloatTensor(points), torch.FloatTensor(landmarks), torch.FloatTensor(heatmaps)
    
    def normalize_points_and_landmarks(self, points, landmarks):
        """포인트 클라우드와 랜드마크를 함께 정규화 (중심 이동만, 스케일링 제거)"""
        # 포인트 클라우드 중심 계산
        centroid = np.mean(points, axis=0)
        
        # 포인트 클라우드와 랜드마크 모두 중심 이동만
        points = points - centroid
        landmarks = landmarks - centroid
        
        return points, landmarks
    
    def save_test_dataset_info(self):
        """테스트 데이터셋 정보를 JSON 파일로 저장 (비교실험용)"""
        import json
        import datetime
        
        test_info = {
            'creation_time': datetime.datetime.now().isoformat(),
            'total_test_samples': len(self.indices),
            'test_data_pairs': [],
            'split_info': {
                'train_samples': int(0.8 * len(self.data_pairs)),
                'test_samples': len(self.indices),
                'split_ratio': '80:20'
            }
        }
        
        # 테스트 데이터 파일 정보 저장
        for idx in self.indices:
            pc_file, landmark_file = self.data_pairs[idx]
            pc_basename = os.path.basename(pc_file)
            landmark_basename = os.path.basename(landmark_file)
            
            # person_id와 expression_id 추출
            person_id = landmark_basename.split('_')[0]
            expression_id = landmark_basename.split('_')[2].split('.')[0]
            
            test_info['test_data_pairs'].append({
                'index': idx,
                'pc_file': pc_file,
                'landmark_file': landmark_file,
                'pc_basename': pc_basename,
                'landmark_basename': landmark_basename,
                'person_id': person_id,
                'expression_id': expression_id
            })
        
        # JSON 파일로 저장
        os.makedirs('./test_dataset_info', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        info_file = f'./test_dataset_info/stage2_test_dataset_{timestamp}.json'
        
        with open(info_file, 'w') as f:
            json.dump(test_info, f, indent=2, ensure_ascii=False)
        
        print(f"테스트 데이터셋 정보가 저장되었습니다: {info_file}")
        
        # 간단한 텍스트 파일도 저장 (가독성용)
        txt_file = f'./test_dataset_info/stage2_test_files_{timestamp}.txt'
        with open(txt_file, 'w') as f:
            f.write("# Stage 2 Test Dataset Files\n")
            f.write(f"# Created: {test_info['creation_time']}\n")
            f.write(f"# Total test samples: {len(self.indices)}\n\n")
            
            for i, pair_info in enumerate(test_info['test_data_pairs']):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Person ID: {pair_info['person_id']}\n")
                f.write(f"  Expression: {pair_info['expression_id']}\n")
                f.write(f"  PC File: {pair_info['pc_basename']}\n")
                f.write(f"  Landmark File: {pair_info['landmark_basename']}\n")
                f.write(f"  Full PC Path: {pair_info['pc_file']}\n")
                f.write(f"  Full Landmark Path: {pair_info['landmark_file']}\n\n")
        
        print(f"테스트 데이터셋 파일 목록이 저장되었습니다: {txt_file}")
    
    def normalize_points(self, points):
        """포인트 클라우드를 정규화 (기존 함수, 호환성 유지)"""
        # 중심을 원점으로 이동
        centroid = np.mean(points, axis=0)
        points = points - centroid
        
        # 스케일 정규화 (최대 거리를 1로)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        points = points / max_dist
        
        return points
    
    def generate_heatmaps(self, points, landmarks):
        """각 랜드마크에 대한 3D Gaussian heatmap 생성"""
        num_landmarks = len(landmarks)
        num_points = len(points)
        heatmaps = np.zeros((num_landmarks, num_points))
        
        for i, landmark in enumerate(landmarks):
            # 각 포인트와 랜드마크 간의 거리 계산
            distances = np.linalg.norm(points - landmark, axis=1)
            
            # Gaussian heatmap 생성
            heatmap = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
            heatmaps[i] = heatmap
        
        return heatmaps
    
    def get_landmark_num(self):
        """랜드마크 개수 반환"""
        if len(self.data_pairs) > 0:
            sample_landmarks = np.load(self.data_pairs[0][1])  # landmark_file
            return len(sample_landmarks)
        return 68  # 기본값

def custom_collate_fn_stage2(batch):
    """Stage 2용 custom collate function"""
    points_list = []
    landmarks_list = []
    heatmaps_list = []
    
    for points, landmarks, heatmaps in batch:
        points_list.append(points)
        landmarks_list.append(landmarks)
        heatmaps_list.append(heatmaps)
    
    # 배치로 스택
    points_batch = torch.stack(points_list, dim=0)  # (B, N, 3)
    landmarks_batch = torch.stack(landmarks_list, dim=0)  # (B, L, 3)
    heatmaps_batch = torch.stack(heatmaps_list, dim=0)  # (B, L, N)
    
    return points_batch, landmarks_batch, heatmaps_batch

def load_test_dataset_from_file(json_file_path, num_points=8192):
    """
    저장된 테스트 데이터셋 JSON 파일에서 테스트 데이터를 로드
    비교실험 시 동일한 테스트 데이터를 사용하기 위함
    """
    import json
    
    with open(json_file_path, 'r') as f:
        test_info = json.load(f)
    
    print(f"Loading test dataset from: {json_file_path}")
    print(f"Test dataset created: {test_info['creation_time']}")
    print(f"Total test samples: {test_info['total_test_samples']}")
    
    # 커스텀 데이터셋 클래스
    class FixedTestDataset:
        def __init__(self, test_data_pairs, num_points):
            self.test_data_pairs = test_data_pairs
            self.num_points = num_points
            self.sigma = 1.0  # heatmap generation
            
        def __len__(self):
            return len(self.test_data_pairs)
            
        def __getitem__(self, idx):
            pair_info = self.test_data_pairs[idx]
            pc_file = pair_info['pc_file']
            landmark_file = pair_info['landmark_file']
            
            # 데이터 로드 및 전처리 (FaceLandmarkDataStage2와 동일)
            points = np.load(pc_file)
            landmarks = np.load(landmark_file)
            
            # 포인트 수 조정
            if len(points) >= self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=False)
            else:
                indices = np.random.choice(len(points), self.num_points, replace=True)
            points = points[indices]
            
            # 정규화 (중심 이동만)
            centroid = np.mean(points, axis=0)
            points = points - centroid
            landmarks = landmarks - centroid
            
            # 히트맵 생성
            num_landmarks = len(landmarks)
            heatmaps = np.zeros((num_landmarks, self.num_points))
            
            for i, landmark in enumerate(landmarks):
                distances = np.linalg.norm(points - landmark, axis=1)
                heatmap = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
                heatmaps[i] = heatmap
            
            return torch.FloatTensor(points), torch.FloatTensor(landmarks), torch.FloatTensor(heatmaps)
    
    # 고정된 테스트 데이터셋 생성
    test_dataset = FixedTestDataset(test_info['test_data_pairs'], num_points)
    
    return test_dataset, test_info 