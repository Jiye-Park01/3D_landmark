import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob


def load_face_data(data):
    Heat_data_sample = np.load('./%s-npy/Heat_data_sample.npy' % data, allow_pickle=True)
    Shape_sample = np.load('./%s-npy/shape_sample.npy' % data, allow_pickle=True)
    landmark_position_select_all = np.load('./%s-npy/landmark_position_select_all.npy' % data, allow_pickle=True)
    if data == 'BU-3DFE' or data == 'FaceScape' or data == 'FRGC':
        return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, data_dir, partition='trainval', num_points=2048):
        self.data_dir = data_dir
        self.partition = partition
        self.num_points = num_points
        self.indices = []  # Initialize indices as empty list
        
        print(f"DEBUG: data_dir received: {data_dir}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        
        print(f"Loading data from: {data_dir}")
        
        # Get all shape files
        self.shape_files = sorted(glob.glob(os.path.join(data_dir, 'shapes', '*_FC_A.npy')))
        print(f"Found {len(self.shape_files)} shape files")
        
        # Get corresponding landmark files by replacing FC_A with FA_A
        self.landmark_files = []
        for shape_file in self.shape_files:
            landmark_file = shape_file.replace('shapes', 'landmarks').replace('FC_A', 'FA_A')
            if os.path.exists(landmark_file):
                self.landmark_files.append(landmark_file)
            else:
                print(f"Warning: Landmark file not found for {shape_file}")
        
        # Remove shape files that don't have corresponding landmark files
        self.shape_files = [f for f in self.shape_files if f.replace('shapes', 'landmarks').replace('FC_A', 'FA_A') in self.landmark_files]
        
        assert len(self.shape_files) == len(self.landmark_files), "Number of shape files and landmark files must match"
        print(f"Found {len(self.shape_files)} matching pairs of shape and landmark files")
        
        if len(self.shape_files) == 0:
            print("ERROR: No matching files found!")
            print(f"Shape files pattern: {os.path.join(data_dir, 'shapes', '*_FC_A.npy')}")
            print(f"Landmark files pattern: {os.path.join(data_dir, 'landmarks', '*_FA_A.npy')}")
            return
        
        # Split into train and val sets (90% train, 10% val)
        num_samples = len(self.shape_files)
        indices = np.random.permutation(num_samples)
        if partition == 'train':
            self.indices = indices[:int(0.9 * num_samples)]
        elif partition == 'val':
            self.indices = indices[int(0.9 * num_samples):]
        else:  # trainval
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        shape_file = self.shape_files[idx]
        landmark_file = self.landmark_files[idx]
        
        # Load point cloud and landmarks
        points = np.load(shape_file)  # (N, 3)
        landmarks = np.load(landmark_file)  # (M, 3)
        
        # 모든 포인트 사용 (샘플링 제거)
        # if len(points) > self.num_points:
        #     indices = np.random.choice(len(points), self.num_points, replace=False)
        #     points = points[indices]
        
        # Convert to torch tensors
        points = torch.from_numpy(points).float()
        landmarks = torch.from_numpy(landmarks).float()
        
        return points, landmarks, shape_file

def custom_collate_fn(batch):
    # Find max number of points in the batch
    max_points = max(points.shape[0] for points, _, _ in batch)
    
    # Initialize padded tensors
    padded_points = []
    landmarks = []
    shape_files = []
    
    for points, landmark, shape_file in batch:
        # Pad points to max_points
        padded_point = torch.zeros(max_points, 3)
        padded_point[:points.shape[0]] = points
        padded_points.append(padded_point)
        landmarks.append(landmark)
        shape_files.append(shape_file)
    
    # Stack all tensors
    padded_points = torch.stack(padded_points)
    landmarks = torch.stack(landmarks)
    
    return padded_points, landmarks, shape_files



