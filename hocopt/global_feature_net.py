"""Pytorch-Geometric implementation of Pointnet++
Original source available at https://github.com/rusty1s/pytorch_geometric"""

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate
import numpy as np
import hocopt.util as util

class Global_Hand_DisNet(torch.nn.Module):
    def __init__(self):
        super(Global_Hand_DisNet, self).__init__()
        HAND_DIS_FEATS = 778 # num_vec * num_bin

        self.hand_lin1 = torch.nn.Linear(HAND_DIS_FEATS, int(HAND_DIS_FEATS/2))
        self.global_hand_lin = torch.nn.Linear(int(HAND_DIS_FEATS/2), 10)
    
    def forward(self, data):
        # soft-argmax
        beta = 10000
        data = F.softmax(data*beta, dim=-1)
        idx = torch.arange(0, 10).to(device='cuda')
        data = data * idx
        data = torch.sum(data, dim=-1)

        x = self.hand_lin1(data) # torch.nn.Linear
        x = self.global_hand_lin(x)

        return F.sigmoid(x)
    
class Global_Object_DisNet(torch.nn.Module):
    def __init__(self):
        super(Global_Object_DisNet, self).__init__()
        OBJ_DIS_FEATS = 2048 # num_sampled_vec * num_bin

        self.obj_lin1 = torch.nn.Linear(OBJ_DIS_FEATS, int(OBJ_DIS_FEATS/2))
        self.global_obj_lin = torch.nn.Linear(int(OBJ_DIS_FEATS/2), 10)
    
    def forward(self, data):
        # soft-argmax
        beta = 10000
        data = F.softmax(data*beta, dim=-1)
        idx = torch.arange(0, 10).to(device='cuda')
        data = data * idx
        data = torch.sum(data, dim=-1)

        x = self.obj_lin1(data) # torch.nn.Linear
        x = self.global_obj_lin(x)

        return F.sigmoid(x)
