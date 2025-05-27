import torch
import numpy as np
from torch.utils.data import Dataset
import os


def load_face_data(data):
    Heat_data_sample = np.load('./%s-npy/Heat_data_sample.npy' % data, allow_pickle=True)
    Shape_sample = np.load('./%s-npy/shape_sample.npy' % data, allow_pickle=True)
    landmark_position_select_all = np.load('./%s-npy/landmark_position_select_all.npy' % data, allow_pickle=True)
    if data == 'BU-3DFE' or data == 'FaceScape' or data == 'FRGC':
        return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, data_dir, partition='trainval'):
        self.data_dir = data_dir
        self.partition = partition
        
        # npy 파일들의 리스트를 가져옴
        self.shape_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'shapes')) if f.endswith('.npy')])
        self.landmark_files = sorted([f for f in os.listdir(os.path.join(data_dir, 'landmarks')) if f.endswith('.npy')])
        
        # 파일 개수 확인
        assert len(self.shape_files) == len(self.landmark_files), "Number of shape files and landmark files must match"
        
    def __getitem__(self, item):
        # 각 파일에서 데이터 로드
        shape = np.load(os.path.join(self.data_dir, 'shapes', self.shape_files[item]))
        landmark = np.load(os.path.join(self.data_dir, 'landmarks', self.landmark_files[item]))
        
        # 텐서로 변환
        shape = torch.Tensor(shape)  # shape: [N, 3] where N is variable
        landmark = torch.Tensor(landmark)  # landmark: [48, 3] (selected landmarks)
        
        if self.partition == 'trainval':
            # 포인트 순서 섞기
            indices = list(range(shape.size()[0]))
            np.random.shuffle(indices)
            shape = shape[indices]
        
        return shape, landmark, None  # heatmap은 모델이 생성

    def __len__(self):
        return len(self.shape_files)




