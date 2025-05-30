'''
@Author: Yuan Wang
@Contact: wangyuan2020@ia.ac.cn
@File: My_args.py
@Time: 2021/12/02 10:55 AM
'''

import argparse
import torch

parser = argparse.ArgumentParser(description='3D Face Landmark Detection')

# base args
parser.add_argument('--exp_name', type=str, default='Face alignment with PAConv', metavar='N', help='Name of the experiment')
parser.add_argument('--model', type=str, default='PAConv', metavar='N', choices=['EdgeConv', 'PAConv'], help='Model to use')
parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing shapes/ and landmarks/ folders')
parser.add_argument('--dataset', type=str, default='custom', help='Dataset name (used for saving models)')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')

# train args
parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size', help='Size of batch)')
parser.add_argument('--epochs', type=int, default=250, metavar='N', help='number of episode to train')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

# optimizor args
parser.add_argument('--loss', type=str, default='mse', metavar='N', choices=['mse', 'adaptive_wing'], help='loss function to use')
parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001, 0.1 if using sgd)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'], help='Scheduler to use, [cos, step]')
parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD', help='the weight decay of optimizor')
parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping threshold')

# data process args
parser.add_argument('--max_threshold', default=10, type=float, help='the maximum threshold of error_rate')
parser.add_argument('--seed', type=int, default=10, metavar='S', help='random seed (default: 1)')
parser.add_argument('--dataset_seed', type=int, default=1, metavar='S', help='train/test dataset random seed (default: 1)')
parser.add_argument('--sigma', type=float, default=10, metavar='Sig', help='Gaussian Variance of heatmap')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=30, metavar='N', help='Num of nearest neighbors to use in PAConv')

# PAConv args
parser.add_argument('--calc_scores', type=str, default='softmax', metavar='cs', help='The way to calculate score')
parser.add_argument('--hidden', type=list, default=[[16], [16], [16], [16]], help='the hidden layers of ScoreNet')
parser.add_argument('--num_matrices', type=list, default=[8, 8, 8, 8], help='the number of weight banks')

# Add num_points and num_landmarks arguments
parser.add_argument('--num_points', type=int, default=2048, help='Number of points to sample from each shape')
parser.add_argument('--num_landmarks', type=int, default=56, help='Number of landmarks')

# CUDA argument
parser.add_argument('--cuda', action='store_true', help='enable CUDA training')
parser.add_argument('--no-cuda', action='store_false', dest='cuda', help='disable CUDA training')
parser.set_defaults(cuda=True) # Default to using CUDA

# Visualization argument
parser.add_argument('--visualize', action='store_true', help='Enable visualization during training')
parser.add_argument('--vis_interval', type=int, default=10, help='Visualization interval in epochs')
parser.set_defaults(visualize=False)  # Default to False

def parse_args():
    return parser.parse_args()

