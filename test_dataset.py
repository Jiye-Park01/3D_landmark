#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

from dataset import FaceLandmarkData

def test_dataset_loading():
    """Test dataset loading with different configurations"""
    
    data_dir = "./dataset"
    
    print("=" * 50)
    print("Testing Dataset Loading")
    print("=" * 50)
    
    # Test 1: A.npy files only
    print("\n1. Testing A.npy files only:")
    try:
        dataset_a = FaceLandmarkData(data_dir=data_dir, num_points=2048, partition='train', use_only_a_files=True)
        print(f"   Successfully loaded {len(dataset_a)} samples (A.npy only)")
        
        # Show first few file names
        print("   First 5 shape files:")
        for i in range(min(5, len(dataset_a.shape_files))):
            print(f"     {os.path.basename(dataset_a.shape_files[i])}")
            
    except Exception as e:
        print(f"   Error loading A.npy dataset: {e}")
    
    # Test 2: All files
    print("\n2. Testing all file types:")
    try:
        dataset_all = FaceLandmarkData(data_dir=data_dir, num_points=2048, partition='train', use_only_a_files=False)
        print(f"   Successfully loaded {len(dataset_all)} samples (all types)")
        
        # Show first few file names
        print("   First 5 shape files:")
        for i in range(min(5, len(dataset_all.shape_files))):
            print(f"     {os.path.basename(dataset_all.shape_files[i])}")
            
    except Exception as e:
        print(f"   Error loading all files dataset: {e}")
    
    # Test 3: Data loading
    print("\n3. Testing data loading:")
    try:
        if len(dataset_a) > 0:
            sample_points, sample_landmarks = dataset_a[0]
            print(f"   Sample points shape: {sample_points.shape}")
            print(f"   Sample landmarks shape: {sample_landmarks.shape}")
            print(f"   Points range: [{sample_points.min():.3f}, {sample_points.max():.3f}]")
            print(f"   Landmarks range: [{sample_landmarks.min():.3f}, {sample_landmarks.max():.3f}]")
    except Exception as e:
        print(f"   Error loading sample data: {e}")
    
    print("\n" + "=" * 50)
    print("Dataset loading test completed!")
    print("=" * 50)

if __name__ == "__main__":
    test_dataset_loading() 