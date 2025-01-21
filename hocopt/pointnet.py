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


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch): # batch: batch*(num_hand_vertices + num_obj_vertices)
        idx = fps(pos, batch, ratio=self.ratio) 
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64) # row:pos[idx] point; col: corresponding to pos[idx]
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn # MLP

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        NUM_FEATS = 25
        NUM_CLASSES = 10

        self.sa1_module = SAModule(0.2, 0.1, MLP([3 + NUM_FEATS, 64, 64, 128])) # TODO, reduce PN params
        self.sa2_module = SAModule(0.25, 0.2, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = SAModule(0.3, 0.25, MLP([256 + 3, 256, 256, 512])) # 
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        self.sa4_module = GlobalSAModule(MLP([512 + 3, 512, 1024, 2048]))

        # self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        # self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        # self.fp1_module = FPModule(3, MLP([128 + NUM_FEATS, 128, 128, 128]))
        self.fp4_module = FPModule(1, MLP([2048 + 512, 512, 512]))
        self.fp3_module = FPModule(3, MLP([512 + 256, 512, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + NUM_FEATS, 128, 128, 128]))

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 128)
        self.lin3 = torch.nn.Linear(128, 128)
        # self.lin3 = torch.nn.Linear(128, NUM_CLASSES)
        self.lin4 = torch.nn.Linear(128, NUM_CLASSES)
        self.lin5 = torch.nn.Linear(128, NUM_CLASSES)

    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch) # len (3)
        sa1_out = self.sa1_module(*sa0_out) # len(3)
        sa2_out = self.sa2_module(*sa1_out) # len(3)
        sa3_out = self.sa3_module(*sa2_out)
        sa4_out = self.sa4_module(*sa3_out)

        # fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        # fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        # x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        fp4_out = self.fp4_module(*sa4_out, *sa3_out)
        fp3_out = self.fp3_module(*fp4_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x) # (90432, 10)
        x = F.dropout(x, p=0.5, training=self.training)
        mid_feature = x.clone()
        x = self.lin4(x)
        weight = 1 + F.softmax(self.lin5(mid_feature), dim=-1)
        x = x * weight

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # bin_weights = torch.Tensor(np.loadtxt(util.AFFORDANCE_BIN_WEIGHTS_FILE)).to(device)
        # x = x * F.sigmoid(weight) * bin_weights # sigmoid() is better than softmax

        # return x
        # return F.sigmoid(x) # big hyperparam, Bound to 0-1
        # print('pre softmax shape', x.shape)
        # return F.log_softmax(x, dim=-1)
        return x, weight.clone()
