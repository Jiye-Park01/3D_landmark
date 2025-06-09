import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
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
        self.num_points = num_points # This will be overridden by max_points_in_dataset
        self.indices = []  # Initialize indices as empty list
        
        print(f"Loading data from: {data_dir}")
        
        # Get all shape files (both FC_A and FC_C)
        self.shape_files = []
        self.shape_files.extend(glob.glob(os.path.join(data_dir, 'shapes', '*_FC_A.npy')))
        self.shape_files.extend(glob.glob(os.path.join(data_dir, 'shapes', '*_FC_C.npy')))
        self.shape_files = sorted(self.shape_files)
        print(f"Found {len(self.shape_files)} shape files")
        
        # Get corresponding landmark files by replacing FC_A/FC_C with FA_A/FA_C
        self.landmark_files = []
        for shape_file in self.shape_files:
            if 'FC_A' in shape_file:
                landmark_file = shape_file.replace('shapes', 'landmarks').replace('FC_A', 'FA_A')
            else:  # FC_C
                landmark_file = shape_file.replace('shapes', 'landmarks').replace('FC_C', 'FA_C')
            
            if os.path.exists(landmark_file):
                self.landmark_files.append(landmark_file)
            else:
                print(f"Warning: Landmark file not found for {shape_file}")
        
        # Remove shape files that don't have corresponding landmark files
        self.shape_files = [f for f in self.shape_files if (
            (f.replace('shapes', 'landmarks').replace('FC_A', 'FA_A') in self.landmark_files) or
            (f.replace('shapes', 'landmarks').replace('FC_C', 'FA_C') in self.landmark_files)
        )]
        
        assert len(self.shape_files) == len(self.landmark_files), "Number of shape files and landmark files must match"
        print(f"Found {len(self.shape_files)} matching pairs of shape and landmark files")
        
        if len(self.shape_files) == 0:
            print("ERROR: No matching files found!")
            print(f"Shape files patterns: {os.path.join(data_dir, 'shapes', '*_FC_A.npy')} and {os.path.join(data_dir, 'shapes', '*_FC_C.npy')}")
            print(f"Landmark files patterns: {os.path.join(data_dir, 'landmarks', '*_FA_A.npy')} and {os.path.join(data_dir, 'landmarks', '*_FA_C.npy')}")
            return
        
        # --- Calculate global max_num_points from the dataset ---
        print("Calculating maximum number of points across the dataset...")
        max_points_in_dataset = 0
        for shape_file_path in self.shape_files:
            try:
                points_data = np.load(shape_file_path)
                if points_data.shape[0] > max_points_in_dataset:
                    max_points_in_dataset = points_data.shape[0]
            except Exception as e:
                print(f"Warning: Could not load {shape_file_path} to determine max points: {e}")
        
        self.num_points = max_points_in_dataset
        print(f"Global maximum points in dataset set to: {self.num_points}")
        # --------------------------------------------------------
        
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
        # Get the index within the filtered lists (self.shape_files, self.landmark_files)
        actual_filtered_idx = self.indices[idx]
        
        # Load point cloud
        points = np.load(self.shape_files[actual_filtered_idx])  # (Original_N, 3)

        original_num_points = points.shape[0] # Store original shape for debug prints
        # print(f"DEBUG: __getitem__ - Item {idx}, Original Shape: {original_num_points}, Global Target num_points: {self.num_points}") # Removed debug print

        if original_num_points < self.num_points:
            # Pad with zeros if fewer points than the global target num_points
            padding_needed = self.num_points - original_num_points
            padding = np.zeros((padding_needed, points.shape[1]), dtype=points.dtype)
            points = np.vstack((points, padding))
            # print(f"DEBUG: __getitem__ - Padded Item {idx}. New Shape: {points.shape}") # Removed debug print
        # else (original_num_points == self.num_points), do nothing.
        # Points with original_num_points > self.num_points should not occur after __init__ modification.

        # Load landmarks
        landmarks = np.load(self.landmark_files[actual_filtered_idx])  # (num_landmarks, 3)

        # Convert to torch tensors
        points = torch.FloatTensor(points)  # (potentially varied_N, 3)
        landmarks = torch.FloatTensor(landmarks)  # (num_landmarks, 3)

        return points, landmarks # Return only points and landmarks, no None

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

def get_dataloader(data_dir, batch_size, num_workers=4, transform=None, partition='train'):
    dataset = FaceLandmarkData(data_dir, partition=partition, num_points=2048) # num_points to match args default
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if partition == 'train' else False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn if 'custom_collate_fn' in globals() else None # Use custom collate if available
    )
    return dataloader



