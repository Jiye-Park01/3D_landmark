'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: util.py
@Time: 2021/12/02 09:59 AM
'''

import numpy as np
import scipy.io as sio
import torch
import h5py
import cv2
import sys
from functools import reduce
import torch.nn.functional as F
from sklearn.metrics import auc
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors

from PointTransformer_model import PointTransformerLandmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device, args):
    model = PointTransformerLandmark(args, args.num_landmarks).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if the loaded checkpoint is a dictionary containing 'model_state_dict'
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loaded model state_dict from full checkpoint.")
    else:
        # Assume it's an old format checkpoint directly containing the model state_dict
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model state_dict from old format checkpoint.")
        
    model.eval()
    return model

def load_shape_data(dataset):
    if dataset == 'BU-3DFE':
        vertics_landmark_name = './BU-3DFE-dataset-mat/vertics_landmark_refine_Read.mat'
        vertics_landmark_all = h5py.File(vertics_landmark_name, 'r')
        shape_all = vertics_landmark_all['shape_all'][0]
        shape_all = [np.array(vertics_landmark_all[vertics_landmark_all['shape_all'][i][0]].value.transpose()) for i in range(len(vertics_landmark_all['shape_all']))]
    elif dataset == 'FRGC':
        vertics_landmark_name = './FRGC-dataset-mat/FRGC_vertics_landmark_Read.mat'
        vertics_landmark_all = h5py.File(vertics_landmark_name, 'r')
        shape_all = vertics_landmark_all['FRGC_shape_all'][0]
        shape_all = [np.array(vertics_landmark_all[vertics_landmark_all['FRGC_shape_all'][i][0]].value.transpose()) for i in range(len(vertics_landmark_all['FRGC_shape_all']))]
    return shape_all

def load_landmark_index(dataset):
    if dataset == 'BU-3DFE':
        landmark_index_name = './BU-3DFE-dataset-mat/landmark_select_refine.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['landmark_index_select_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    elif dataset == 'FRGC':
        landmark_index_name = './FRGC-dataset-mat/FRGC_landmark_select.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['FRGC_landmark_index_select_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    else:
        landmark_index_name = './FaceScape-publish-dataset-mat/FaceScape_landmark_select.mat'
        landmark_index_all = sio.loadmat(landmark_index_name)
        landmark_index_select_all = landmark_index_all['FaceScape_landmark_index_all'][0]
        landmark_index_select_all = [np.array(landmark_index_select_all[k]) for k in range(len(landmark_index_select_all))]
    return landmark_index_select_all


def load_landmark_position(dataset):
    if dataset == 'BU-3DFE':
        landmark_position_name = './BU-3DFE-dataset-mat/landmark_select_refine.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_select_all = landmark_position_all['landmark_position_select_all'][0]
        landmark_position_select_all = [np.array(landmark_position_select_all[k]) for k in range(len(landmark_position_select_all))] 
    elif dataset == 'FRGC':
        landmark_position_name = './FRGC-dataset-mat/FRGC_landmark_select.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_select_all = landmark_position_all['FRGC_landmark_position_select_all'][0]
        landmark_position_select_all = [np.array(landmark_position_select_all[k]) for k in range(len(landmark_position_select_all))] 
    else:
        landmark_position_name = './FaceScape-publish-dataset-mat/FaceScape_landmark_select.mat'
        landmark_position_all = sio.loadmat(landmark_position_name)
        landmark_position_select_all = landmark_position_all['FaceScape_landmark_position_all'][0]
        landmark_position_select_all = [np.array(landmark_position_select_all[k]) for k in range(len(landmark_position_select_all))]  
    return landmark_position_select_all


def load_Heatmap_data():
    Heat_data_all = np.load('Heat_data_all.npy', allow_pickle=True)
    return Heat_data_all


def predict_landmarks(model, points, device):
    model.eval()
    with torch.no_grad():
        # Shape 보정
        if points.ndim == 2:
            points = points.unsqueeze(0)
        elif points.ndim == 3 and points.shape[1] == 3:
            points = points.transpose(1, 2)

        points = points.to(device)  # (B, N, 3)
        pred_heatmap = model(points)  # (B, L, N)
        pred_coords = heatmap_to_coordinates(pred_heatmap, points)
    return pred_coords


def heatmap_to_coordinates(heatmap, points):
    # heatmap: (B, L, N), points: (B, N, 3)
    B, L, N = heatmap.shape
    coords = []
    for b in range(B):
        landmarks = []
        for l in range(L):
            idx = torch.argmax(heatmap[b, l])
            landmarks.append(points[b, idx])
        coords.append(torch.stack(landmarks))  # (L, 3)
    return torch.stack(coords)  # (B, L, 3)



def calculateHeatMap_Euclidean(shape_all, landmark_position_select_all, sigma):
    Heat_data_all = []
    for i in range(len(shape_all)):
        shape_i = shape_all[i].reshape(shape_all[i].shape[0], 1, shape_all[i].shape[1]).repeat(landmark_position_select_all[i].shape[0], axis=1)
        Euclidean_distance_i = np.linalg.norm((shape_i - landmark_position_select_all[i]), axis=2)
        Heat_data_i = Gaussian_Heatmap(Euclidean_distance_i, sigma)
        Heat_data_all.append(Heat_data_i)
    return Heat_data_all


def Gaussian_Heatmap(Distance, sigma):
    D2 = Distance * Distance
    S2 = 2.0 * sigma * sigma
    Exponent = D2 / S2
    heatmap = np.exp(-Exponent)
    return heatmap


def compute_sample_index(Heat_data, number_point, landmark_index, rand_seed):
    np.random.seed(rand_seed)
    point_num = np.array(Heat_data).shape[0]
    index_1 = np.arange(point_num)
    index = np.random.choice(index_1, size=number_point, replace=False)
    return index


def get_dists(points1, points2):
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists)  # Very Important for dist = 0.
    return torch.sqrt(dists).float()


'''
Farthest Point Sampling from 
Qi C R, Yi L, Su H, et al. Pointnet++: Deep hierarchical feature learning on point sets in a metric space. NIPS 2017.
The following module is based on https://github.com/erikwijmans/Pointnet2_PyTorch
'''
def fps(xyz, M):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(size=(B, M), dtype=torch.long).to(device)
    dists = torch.ones(B, N).to(device) * 1e5
    inds = torch.randint(0, N, size=(B, ), dtype=torch.long).to(device)
    batchlists = torch.arange(0, B, dtype=torch.long).to(device)
    for i in range(M):
        centroids[:, i] = inds
        cur_point = xyz[batchlists, inds, :] 
        cur_dist = torch.squeeze(get_dists(torch.unsqueeze(cur_point, 1), xyz), dim=1)
        dists[cur_dist < dists] = cur_dist[cur_dist < dists]
        inds = torch.max(dists, dim=1)[1]
    return centroids


def random_sample(shape_all, Heat_data_all, number_point, rand_seed, sample_way, dataset):    
    print('Start load landmark index data! ')
    landmark_index_select_all = load_landmark_index(dataset)
    print('Load landmark index data successfully')
    if sample_way == 'Random':
        np.random.seed(rand_seed)
        random_matrix = [compute_sample_index(Heat_data_all[i], number_point, landmark_index_select_all[i]-1, rand_seed) for i in range(len(Heat_data_all))]
        print('Finish the caculation of random matrix')
        Heat_data_sample = [np.array(Heat_data_all[j])[random_matrix[j], :] for j in range(len(Heat_data_all))]
        shape_sample = [np.array(shape_all[j])[random_matrix[j], :] for j in range(len(shape_all))]
    elif sample_way == 'FPS':
        FPS_matrix = [fps(torch.from_numpy(shape_all[i]).unsqueeze(0).to(device), number_point) for i in range(len(Heat_data_all))]
        print('Finish the caculation of random matrix')
        Heat_data_sample = [np.array(Heat_data_all[j])[FPS_matrix[j].squeeze(0).cpu(), :] for j in range(len(Heat_data_all))]
        shape_sample = [np.array(shape_all[j])[FPS_matrix[j].squeeze(0).cpu(), :] for j in range(len(shape_all))]
    else:
        raise AssertionError('Invalid Sample Way')
    return Heat_data_sample, shape_sample


def soft_argmax(Heatmap, point, alpha):
    Heatmap = Heatmap * alpha
    soft_max = F.softmax(Heatmap, dim=2)
    indices_kernel = torch.arange(start=0, end=point.size(2), device=device).float()
    conv = soft_max * indices_kernel
    landmark_index_pred = conv.sum(2).floor().type_as(indices_kernel)
    landmark_coords_pred = [point[i, :, landmark_index_pred[i].long()].unsqueeze(0) for i in range(point.size(0))]
    landmark_coords_pred = torch.cat(landmark_coords_pred, dim=0)
    return landmark_coords_pred.permute(0, 2, 1)


def My_MDS(D, d=2):
    DSquare = D
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5 * (DSquare[i][j] - rowMean[i] - columnMean[j] + totalMean)
    eigVal, eigVec = np.linalg.eig(B)
    eigValSorted_indices = np.argsort(eigVal)
    topd_eigVec = eigVec[:,eigValSorted_indices[:-d-1:-1]] 
    X = np.dot(topd_eigVec, np.sqrt(np.diag(eigVal[:-d-1:-1])))
    return X

def landmark_regression(shape, Heatmap, regression_point_num=10, device=None):
    """
    shape: (N, 3) torch tensor
    Heatmap: (N, L) torch tensor
    return: (1, L, 3) torch tensor
    """
    N, L = Heatmap.shape
    if device is None:
        device = shape.device

    # 1. 각 랜드마크별로 heatmap 값이 큰 r개 포인트 선택
    topk_vals, topk_idx = torch.topk(Heatmap, regression_point_num, dim=0)  # (r, L)
    # shape_sort_select: (L, r, 3)
    shape_sort_select = shape[topk_idx]  # (r, L, 3)
    shape_sort_select = shape_sort_select.permute(1, 0, 2)  # (L, r, 3)
    Heatmap_sort_select = topk_vals.permute(1, 0).unsqueeze(-1)  # (L, r, 1)

    # 2. 거리 행렬 계산 (L, r, r)
    diff = shape_sort_select.unsqueeze(2) - shape_sort_select.unsqueeze(1)  # (L, r, r, 3)
    D_Matrix = torch.norm(diff, dim=-1)  # (L, r, r)

    # 3. MDS 근사 unfolding (고전적 MDS: 중심화 + SVD)
    # 중심화
    J = torch.eye(regression_point_num, device=device) - 1.0 / regression_point_num
    J = J.unsqueeze(0).expand(L, -1, -1)  # (L, r, r)
    B = -0.5 * J @ (D_Matrix ** 2) @ J  # (L, r, r)
    # SVD로 2D 좌표 추출
    U, S, V = torch.linalg.svd(B)
    shape_MDS = U[:, :, :2] * S[:, :2].sqrt().unsqueeze(1)  # (L, r, 2)
    # 3D로 확장 (z=0)
    shape_MDS = torch.cat([shape_MDS, torch.zeros(L, regression_point_num, 1, device=device)], dim=2)  # (L, r, 3)

    # 4. weighted 평균 (unfolded landmark 좌표)
    weights = Heatmap_sort_select / (Heatmap_sort_select.sum(1, keepdim=True) + 1e-8)  # (L, r, 1)
    landmark2D = (weights * shape_MDS).sum(1)  # (L, 3)

    # 5. 반환 shape 맞추기
    return landmark2D.unsqueeze(0)  # (1, L, 3)

def get_rigid(src, dst):
    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    H = reduce(lambda s, p: s + np.outer(p[0], p[1]), zip(src - src_mean, dst - dst_mean), np.zeros((3,3)))
    u, s, v = np.linalg.svd(H)
    R = v.T.dot(u.T)
    T = - R.dot(src_mean) + dst_mean
    return np.hstack((R, T[:, np.newaxis]))

def get_3D_FAN_NME(pred_landmark, gt_landmark):
    NME_single = torch.sum(torch.norm(pred_landmark - gt_landmark, dim=2), 0)
    NME = torch.mean(NME_single)
    return NME, NME_single

def calc_error_rate_i(pred_landmark_coords, gt_landmark_coords):
    error_single = torch.norm(pred_landmark_coords - gt_landmark_coords, dim=1)
    error = torch.mean(error_single)
    return error, error_single

def main_sample(number_point, seed, sigma, sample_way, dataset):
    # load point clouds of faces
    print('Start load shape data ! ')
    shape_all = load_shape_data(dataset)
    print('Load shape data successfully')
    # load landmark position of faces
    print('Start load landmark position data ! ')
    landmark_position_select_all = load_landmark_position(dataset)
    print('Load landmark position data successfully')
    # compute the distance(Geodesic or Euclidean distance) from landmarks to all points
    print('Start calculate and save all Heatmaps !')
    Heat_data_all = calculateHeatMap_Euclidean(shape_all, landmark_position_select_all, sigma)
    print('Calculate and save all Heatmaps successfully')
    Heat_data_sample, shape_sample, = random_sample(shape_all, Heat_data_all, number_point, seed, sample_way, dataset)
    np.save('./%s-npy/Heat_data_sample.npy' % dataset, Heat_data_sample)
    np.save('./%s-npy/shape_sample.npy' % dataset, shape_sample)
    np.save('./%s-npy/landmark_position_select_all.npy' % dataset, landmark_position_select_all)
