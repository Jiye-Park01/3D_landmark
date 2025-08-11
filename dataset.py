import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import glob

from augmentations import normalize_data


def load_face_data(data):
    Heat_data_sample = np.load('./%s-npy/Heat_data_sample.npy' % data, allow_pickle=True)
    Shape_sample = np.load('./%s-npy/shape_sample.npy' % data, allow_pickle=True)
    landmark_position_select_all = np.load('./%s-npy/landmark_position_select_all.npy' % data, allow_pickle=True)
    if data == 'BU-3DFE' or data == 'FaceScape' or data == 'FRGC':
        return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, data_dir, partition='trainval', num_points=2048, use_only_a_files=True):
        self.data_dir = data_dir
        self.partition = partition
        self.num_points = num_points # This will be overridden by max_points_in_dataset
        self.indices = []  # Initialize indices as empty list
        self.use_only_a_files = use_only_a_files
        
        print(f"Loading data from: {data_dir}")
        print(f"Using only A.npy files: {use_only_a_files}")
        
        if use_only_a_files:
            # Get only A.npy shape files (FC_A only)
            self.shape_files = []
            self.shape_files.extend(glob.glob(os.path.join(data_dir, 'shapes', f'*_FC_A_vertices.npy')))
            self.shape_files = sorted(self.shape_files)
            print(f"Found {len(self.shape_files)} A.npy shape files (FC_A only)")
        else:
            # Get all shape files (FC_A, FC_B, FC_C, FC_F, FC_K)
            self.shape_files = []
            for suffix in ['A', 'B', 'C', 'F', 'K']:
                self.shape_files.extend(glob.glob(os.path.join(data_dir, 'shapes', f'*_FC_{suffix}.npy')))
            self.shape_files = sorted(self.shape_files)
            print(f"Found {len(self.shape_files)} shape files (all types)")
        
        if use_only_a_files:
            # Get corresponding landmark files by replacing FC_A with FC_A_landmarks
            self.landmark_files = []
            for shape_file in self.shape_files:
                landmark_file = shape_file.replace('shapes', 'landmarks').replace('FC_A_vertices.npy', 'FC_A_landmarks.npy')
                if landmark_file and os.path.exists(landmark_file):
                    self.landmark_files.append(landmark_file)
                else:
                    print(f"Warning: Landmark file not found for {shape_file}")
            
            # Remove shape files that don't have corresponding landmark files
            valid_shape_files = []
            for f in self.shape_files:
                if f.replace('shapes', 'landmarks').replace('FC_A_vertices.npy', 'FC_A_landmarks.npy') in self.landmark_files:
                    valid_shape_files.append(f)
            self.shape_files = valid_shape_files
        else:
            # Get corresponding landmark files by replacing FC_* with FC_*_landmarks
            self.landmark_files = []
            for shape_file in self.shape_files:
                for suffix in ['A', 'B', 'C', 'F', 'K']:
                    if f'FC_{suffix}_vertices.npy' in shape_file:
                        landmark_file = shape_file.replace('shapes', 'landmarks').replace(f'FC_{suffix}_vertices.npy', f'FC_{suffix}_landmarks.npy')
                        break
                else:
                    landmark_file = None
                if landmark_file and os.path.exists(landmark_file):
                    self.landmark_files.append(landmark_file)
                else:
                    print(f"Warning: Landmark file not found for {shape_file}")
            
            # Remove shape files that don't have corresponding landmark files
            valid_shape_files = []
            for f in self.shape_files:
                for suffix in ['A', 'B', 'C', 'F', 'K']:
                    if f.replace('shapes', 'landmarks').replace(f'FC_{suffix}_vertices.npy', f'FC_{suffix}_landmarks.npy') in self.landmark_files:
                        valid_shape_files.append(f)
                        break
            self.shape_files = valid_shape_files
        
        assert len(self.shape_files) == len(self.landmark_files), "Number of shape files and landmark files must match"
        print(f"Found {len(self.shape_files)} matching pairs of shape and landmark files")
        
        if len(self.shape_files) == 0:
            print("ERROR: No matching files found!")
            if use_only_a_files:
                print("Shape files pattern:", os.path.join(data_dir, 'shapes', '*_FC_A_vertices.npy'))
                print("Landmark files pattern:", os.path.join(data_dir, 'landmarks', '*_FC_A_landmarks.npy'))
            else:
                print("Shape files patterns:", [os.path.join(data_dir, 'shapes', f'*_FC_{s}_vertices.npy') for s in ['A','B','C','F','K']])
                print("Landmark files patterns:", [os.path.join(data_dir, 'landmarks', f'*_FC_{s}_landmarks.npy') for s in ['A','B','C','F','K']])
            return
        
        # --- Calculate global max_num_points from the dataset ---
        print("Calculating maximum number of points across the dataset...")
        max_points_in_dataset = 0
        # Also initialize min/max bounds for each landmark in normalized space
        # Assuming num_landmarks is consistent (57 in train3.py)
        # Initialize with +/- inf to capture true min/max
        # Shape: (num_landmarks, num_dims * 2) -> (57, 6) for (min_x, max_x, min_y, max_y, min_z, max_z)
        # We need to know num_landmarks at this stage. Let's assume it's passed or derived.
        # For now, we'll get it from the first landmark file or from args.
        # Assuming 68 landmarks based on the actual data
        num_landmarks_val = 68 # Hardcode for now, will refine if necessary

        # Initialize ranges with extreme values
        self.landmark_ranges_min = np.full((num_landmarks_val, 3), np.inf)
        self.landmark_ranges_max = np.full((num_landmarks_val, 3), -np.inf)


        for i, shape_file_path in enumerate(self.shape_files):
            try:
                points_data = np.load(shape_file_path)
                if points_data.shape[0] > max_points_in_dataset:
                    max_points_in_dataset = points_data.shape[0]
                
                # Load and normalize landmarks for range calculation
                landmarks_data = np.load(self.landmark_files[i])
                
                # Convert to torch tensor for normalization
                points_tensor = torch.from_numpy(points_data).float().unsqueeze(0) # Add batch dim
                landmarks_tensor = torch.from_numpy(landmarks_data).float().unsqueeze(0) # Add batch dim

                # Normalize both points and landmarks using the same transformation
                normalized_points_tensor, normalized_landmarks_tensor = normalize_data(points_tensor, landmarks_tensor)
                
                # Update min/max for each landmark based on normalized data
                self.landmark_ranges_min = np.minimum(self.landmark_ranges_min, normalized_landmarks_tensor.squeeze(0).cpu().numpy())
                self.landmark_ranges_max = np.maximum(self.landmark_ranges_max, normalized_landmarks_tensor.squeeze(0).cpu().numpy())

            except Exception as e:
                print(f"Warning: Could not load or normalize {shape_file_path} or its landmarks to determine ranges: {e}")
        
        self.num_points = max_points_in_dataset
        print(f"Global maximum points in dataset set to: {self.num_points}")
        print(f"Calculated landmark movement ranges (min/max X, Y, Z) in normalized space.")
        
        # Debugging: Print calculated landmark ranges
        print(f"DEBUG: landmark_ranges_min: {self.landmark_ranges_min.min():.4f} to {self.landmark_ranges_min.max():.4f}")
        print(f"DEBUG: landmark_ranges_max: {self.landmark_ranges_max.min():.4f} to {self.landmark_ranges_max.max():.4f}")
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
    # batch: list of (points, landmark, file_path)
    invalid_files = []
    filtered = []
    for item in batch:
        # item: (points, landmark, file_path or None)
        if item[1].shape[0] == 68:
            filtered.append(item)
        else:
            # 파일 경로가 있으면 기록
            if len(item) > 2 and item[2] is not None:
                invalid_files.append(str(item[2]))
            else:
                invalid_files.append("unknown_file")
    if invalid_files:
        with open("invalid_landmark_files.txt", "a") as f:
            for fname in invalid_files:
                f.write(fname + "\n")
        print(f"[WARNING] Batch contains landmark shape != 68. Invalid files: {invalid_files}")
    if len(filtered) == 0:
        raise ValueError("No valid samples with 68 landmarks in this batch!")
    points_list = [item[0] for item in filtered]
    landmarks_list = [item[1] for item in filtered]
    points = torch.stack(points_list)
    landmarks = torch.stack(landmarks_list)
    return points, landmarks, None

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



