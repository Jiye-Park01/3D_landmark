import torch
import torch.nn as nn
import torch.nn.functional as F
from My_args import *
# from PAConv.util.PAConv_util import knn, get_graph_feature, get_scorenet_input, feat_trans_dgcnn, ScoreNet, Attention_Layer
from PAConv.part_seg.util.PAConv_util import knn, get_scorenet_input, feat_trans_dgcnn, ScoreNet

# Import get_graph_feature from scene_seg/util/paconv_util.py (where you found it before)
from PAConv.scene_seg.util.paconv_util import get_graph_feature
from PAConv.obj_cls.cuda_lib.functional import assign_score_withk as assemble_dgcnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
PAConv model from 
Xu M, Ding R, Zhao H, et al. PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds. CVPR2021
The following module is based on https://github.com/CVMI-Lab/PAConv
'''

class HeatmapHead(nn.Module):
    def __init__(self, in_channels, num_landmarks, sigma=10):
        super(HeatmapHead, self).__init__()
        self.num_landmarks = num_landmarks
        self.sigma = sigma
        
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, num_landmarks, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class PAConv(nn.Module):
    def __init__(self, args, landmark_num):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.k
        self.landmark_num = landmark_num
        self.calc_scores = args.calc_scores
        self.hidden = args.hidden

        # PAConv layers
        self.m2, self.m3, self.m4, self.m5 = args.num_matrices
        self.scorenet2 = ScoreNet(10, self.m2, hidden_unit=self.hidden[0])
        self.scorenet3 = ScoreNet(10, self.m3, hidden_unit=self.hidden[1])
        self.scorenet4 = ScoreNet(10, self.m4, hidden_unit=self.hidden[2])
        self.scorenet5 = ScoreNet(10, self.m5, hidden_unit=self.hidden[3])

        # Feature dimensions
        i2 = 64
        o2 = i3 = 64
        o3 = i4 = 64
        o4 = i5 = 64
        o5 = 64

        # Initialize weight matrices
        tensor2 = nn.init.kaiming_normal_(torch.empty(self.m2, i2 * 2, o2), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i2 * 2, self.m2 * o2)
        tensor3 = nn.init.kaiming_normal_(torch.empty(self.m3, i3 * 2, o3), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i3 * 2, self.m3 * o3)
        tensor4 = nn.init.kaiming_normal_(torch.empty(self.m4, i4 * 2, o4), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i4 * 2, self.m4 * o4)
        tensor5 = nn.init.kaiming_normal_(torch.empty(self.m5, i5 * 2, o5), nonlinearity='relu') \
            .permute(1, 0, 2).contiguous().view(i5 * 2, self.m5 * o5)

        self.matrice2 = nn.Parameter(tensor2, requires_grad=True)
        self.matrice3 = nn.Parameter(tensor3, requires_grad=True)
        self.matrice4 = nn.Parameter(tensor4, requires_grad=True)
        self.matrice5 = nn.Parameter(tensor5, requires_grad=True)

        # Batch normalization layers
        self.bn2 = nn.BatchNorm1d(o2, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(o3, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(o4, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(o5, momentum=0.1)
        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)

        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                 nn.BatchNorm2d(64, momentum=0.1))
        
        # Additional convolutional layers
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 256, 1)
        self.conv5 = nn.Conv1d(256, 512, 1)
        self.conv6 = nn.Conv1d(1024, 1024, 1)
        self.conv7 = nn.Conv1d(3008, 1024, 1)
        
        # Transform networks
        self.transform_net2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.transform_net3 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.transform_net4 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.transform_net5 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Heatmap head
        self.heatmap_head = HeatmapHead(1024, landmark_num, sigma=args.sigma)

    def forward(self, x):
        # x: (B, 3, N) where N is variable
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Get KNN indices first
        idx = knn(x.permute(0, 2, 1), k=self.k)  # (B, N, k)
        
        # Get graph feature using the computed indices
        x = get_graph_feature(x, k=self.k, idx=idx)  # (B, N, k, 6)
        
        # Layer 1
        x = x.permute(0, 3, 1, 2)  # (B, 6, N, k)
        x = self.conv1(x)  # (B, 64, N, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (B, 64, N)
        
        # Layer 2
        x2 = self.conv2(x)  # (B, 64, N)
        x2_transformed = self.transform_net2(x2)  # (B, 64, N)
        x2 = x2_transformed  # Skip matrix multiplication for now
        
        # Layer 3
        x3 = self.conv3(x2)  # (B, 128, N)
        x3_transformed = self.transform_net3(x3)  # (B, 128, N)
        x3 = x3_transformed  # Skip matrix multiplication for now
        
        # Layer 4
        x4 = self.conv4(x3)  # (B, 256, N)
        x4_transformed = self.transform_net4(x4)  # (B, 256, N)
        x4 = x4_transformed  # Skip matrix multiplication for now
        
        # Layer 5
        x5 = self.conv5(x4)  # (B, 512, N)
        x5_transformed = self.transform_net5(x5)  # (B, 512, N)
        x5 = x5_transformed  # Skip matrix multiplication for now
        
        # Global feature
        x = torch.cat([x, x2, x3, x4, x5], dim=1)  # (B, 960, N)
        x = self.conv6(x)  # (B, 1024, N)
        x = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        x = x.repeat(1, 1, num_points)  # (B, 1024, N)
        
        # Final feature
        x = torch.cat([x, x, x2, x3, x4, x5], dim=1)  # (B, 1984, N)
        x = self.conv7(x)  # (B, 1024, N)
        
        # Heatmap prediction
        heatmap = self.heatmap_head(x)  # (B, num_landmarks, N)
        
        return heatmap


