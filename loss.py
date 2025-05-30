import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Adaptive Wing Loss from 
Wang X, Bo L, Fuxin L. Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression. ICCV2019.
The following module is based on https://github.com/protossw512/AdaptiveWingLoss
'''
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        """
        pred: (B, L, N) predicted heatmap
        target: (B, L, N) target heatmap
        """
        delta_y = torch.abs(target - pred)
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        
        y1 = target[delta_y < self.theta]
        y2 = target[delta_y >= self.theta]
        
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - 1))
        loss2 = self.omega * (delta_y2 - self.theta/2)
        
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class HeatmapLoss(nn.Module):
    def __init__(self, sigma=2.0):
        super(HeatmapLoss, self).__init__()
        self.sigma = sigma
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred: (B, L, N) predicted heatmap
        target: (B, L, N) target heatmap
        """
        return self.criterion(pred, target)


def generate_heatmap(points, landmarks, sigma=2.0):
    """
    Generates Gaussian heatmap for landmarks on the point cloud.
    Args:
        points (torch.Tensor): Point cloud data (B, N, 3)
        landmarks (torch.Tensor): Landmark data (B, L, 3)
        sigma (float): Standard deviation for Gaussian kernel.
    Returns:
        torch.Tensor: Heatmap (B, L, N)
    """
    B, N, _ = points.shape
    _, L, _ = landmarks.shape
    
    # Reshape points for broadcasting (B, 1, N, 3)
    points = points.unsqueeze(1)
    # Reshape landmarks for broadcasting (B, L, 1, 3)
    landmarks = landmarks.unsqueeze(2)
    
    # Calculate squared Euclidean distance (B, L, N)
    dist_sq = torch.sum((points - landmarks)**2, dim=-1)
    
    # Calculate heatmap using Gaussian kernel
    heatmap = torch.exp(-dist_sq / (2 * sigma**2))
    
    return heatmap


def get_predicted_landmarks_from_heatmap(points, pred_heatmap):
    """
    Extracts 3D landmark positions from predicted heatmap.
    Finds the point with the maximum heatmap value for each landmark.
    Args:
        points (torch.Tensor): Point cloud data (B, N, 3)
        pred_heatmap (torch.Tensor): Predicted heatmap (B, L, N)
    Returns:
        torch.Tensor: Predicted landmark positions (B, L, 3)
    """
    B, L, N = pred_heatmap.shape
    pred_landmarks = torch.zeros(B, L, 3).to(points.device)
    
    for b in range(B):
        for l in range(L):
            # Find the index of the point with the maximum heatmap value for landmark l
            max_idx = torch.argmax(pred_heatmap[b, l])
            # Get the 3D coordinate of this point
            pred_landmarks[b, l] = points[b, max_idx, :]
            
    return pred_landmarks

