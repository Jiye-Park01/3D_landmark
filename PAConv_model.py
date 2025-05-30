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

        # Batch normalization layers: 채널 수 = o (64)
        self.bn2 = nn.BatchNorm1d(o2, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(o3, momentum=0.1)
        self.bn4 = nn.BatchNorm1d(o4, momentum=0.1)
        self.bn5 = nn.BatchNorm1d(o5, momentum=0.1)
        self.bnt = nn.BatchNorm1d(1024, momentum=0.1)

        # Convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=True),
                                 nn.BatchNorm2d(64, momentum=0.1))
        self.convt = nn.Sequential(nn.Conv1d(64*5, 1024, kernel_size=1, bias=False),
                                 self.bnt)

        # Heatmap head
        self.heatmap_head = HeatmapHead(1024, landmark_num, sigma=args.sigma)

    def forward(self, x):
        batch_size = x.size(0)
        num_channels = x.size(1) # Should be 3 for initial points
        num_points = x.size(2)

        # Input x is expected to be (B, 3, N)

        # Compute KNN indices
        idx = knn(x.permute(0, 2, 1), k=self.k) # (batch_size, num_points, k)

        # Compute xyz input for ScoreNet
        xyz = get_scorenet_input(x.permute(0, 2, 1), k=self.k, idx=idx)

        # PAConv Block 1
        x = get_graph_feature(x, k=self.k, idx=idx)  # b, n, k, 2c
        
        # Fix tensor dimensions for conv1
        # Convert from (b, n, k, 2c) to (b, 2c, n, k)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x)) # b, 64, n, k
        x1 = x.max(dim=-1, keepdim=False)[0] # b, 64, n

        # Layer 2
        x2_input = x1 # (b, 64, n)
        x2_transformed, center2_transformed = feat_trans_dgcnn(point_input=x2_input, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score2, point_input=x2_transformed, center_input=center2_transformed, knn_idx=idx, aggregate='sum')
        # assemble_dgcnn output is (B, O, N), no need to permute
        x = x.contiguous()  # Ensure memory layout is contiguous
        x2 = F.relu(self.bn2(x)) # b, 64, n

        # Layer 3
        x3_input = x2 # (b, 64, n)
        x3_transformed, center3_transformed = feat_trans_dgcnn(point_input=x3_input, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score3, point_input=x3_transformed, center_input=center3_transformed, knn_idx=idx, aggregate='sum')
        x = x.contiguous()  # Ensure memory layout is contiguous
        x3 = F.relu(self.bn3(x)) # b, 64, n

        # Layer 4
        x4_input = x3 # (b, 64, n)
        x4_transformed, center4_transformed = feat_trans_dgcnn(point_input=x4_input, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score4, point_input=x4_transformed, center_input=center4_transformed, knn_idx=idx, aggregate='sum')
        x = x.contiguous()  # Ensure memory layout is contiguous
        x4 = F.relu(self.bn4(x)) # b, 64, n

        # Layer 5
        x5_input = x4 # (b, 64, n)
        x5_transformed, center5_transformed = feat_trans_dgcnn(point_input=x5_input, kernel=self.matrice5, m=self.m5)
        score5 = self.scorenet5(xyz, calc_scores=self.calc_scores, bias=0)
        x = assemble_dgcnn(score=score5, point_input=x5_transformed, center_input=center5_transformed, knn_idx=idx, aggregate='sum')
        x = x.contiguous()  # Ensure memory layout is contiguous
        x5 = F.relu(self.bn5(x)) # b, 64, n

        # Concatenate features
        xx = torch.cat((x1, x2, x3, x4, x5), dim=1) # b, 64*5=320, n

        # Final convolution
        xc = F.relu(self.convt(xx)) # b, 1024, n

        # Generate heatmaps
        heatmaps = self.heatmap_head(xc) # b, num_landmarks, n
        
        return heatmaps


