# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import hocopt.pointnet as pointnet
from hocopt.global_feature_net import Global_Hand_DisNet, Global_Object_DisNet
import torch.nn.functional as F
from pytorch3d import ops, transforms
import hocopt.util as util
import numpy as np


class CREN(nn.Module):
    def __init__(self, normalize_pts=True, test=False):
        super(CREN, self).__init__()
        self.normalize_pts = normalize_pts
        self.test = test

        self.pointnet = pointnet.Net()
        pointnet_total_params = sum(p.numel() for p in self.pointnet.parameters() if p.requires_grad)
        if not self.test:
            self.hand_dis_net = Global_Hand_DisNet()
            self.obj_dis_net = Global_Object_DisNet()
            hand_dis_net_params = sum(p.numel() for p in self.hand_dis_net.parameters() if p.requires_grad)
            obj_dis_net_params = sum(p.numel() for p in self.obj_dis_net.parameters() if p.requires_grad)

            print('Backbone params:\n pointnet:{}, hand_dis_net:{}, obj_dis_net:{}'.format(pointnet_total_params, hand_dis_net_params, obj_dis_net_params))
        else:
            print('Backbone params:\n pointnet:{}'.format(pointnet_total_params))
        print('The current stage is {}'.format("testing process" if self.test else "training process"))
        
    def forward(self, hand_verts, hand_feats, obj_verts, obj_feats):
        device = hand_verts.device
        batch_size = hand_verts.shape[0]
        out = dict()

        if self.normalize_pts:
            tform = self.get_normalizing_tform(hand_verts, obj_verts)
            hand_verts = util.apply_tform(tform, hand_verts)
            obj_verts = util.apply_tform(tform, obj_verts)
            # util.vis_pointcloud(obj_verts, hand_verts)  # View pointnet input

        x, pos, batch = self.verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats)
        output_net, weight = self.pointnet(x, pos, batch) # (90432, 10); (90432,10) 90432 = 32*778 + 32*2048
        contact = output_net.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)
        contact_weight = weight.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)
        # contact_batched = self.pointnet(x, pos, batch)
        # contact = contact_batched.view(batch_size, hand_verts.shape[1] + obj_verts.shape[1], 10)

        out['contact_hand'] = contact[:, :hand_verts.shape[1], :] # (batch, 778, 10)
        out['contact_obj'] = contact[:, hand_verts.shape[1]:, :] # (batch, 2048, 10)
        if not self.test:
            contact_hand_clone = out['contact_hand'].clone()
            contact_obj_clone = out['contact_obj'].clone()
            out['dis_feat_hand'] = self.hand_dis_net(contact_hand_clone)
            out['dis_feat_obj'] = self.obj_dis_net(contact_obj_clone)
        out['contact_hand_weight'] = contact_weight[:, :hand_verts.shape[1], :]
        out['contact_obj_weight'] = contact_weight[:, hand_verts.shape[1]:, :]
        # w_obj, w_hand = self.get_entropy_weight(out['contact_obj'].clone().detach().cpu().numpy(), out['contact_hand'].clone().detach().cpu().numpy())
        # out['contact_obj'] = out['contact_obj'] * w_obj
        # out['contact_hand'] = out['contact_hand'] * w_hand

        return out

    @staticmethod
    def get_normalizing_tform(hand_verts, obj_verts, random_rot=True):
        """
        Find a 4x4 rigid transform to normalize the pointcloud. We choose the object center of mass to be the origin,
        the hand center of mass to be along the +X direction, and the rotation around this axis to be random.
        :param hand_verts: (batch, 778, 3)
        :param obj_verts: (batch, 2048, 3)
        :return: tform: (batch, 4, 4)
        """
        with torch.no_grad():
            obj_centroid = torch.mean(obj_verts, dim=1)  # (batch, 3)
            hand_centroid = torch.mean(hand_verts, dim=1)

            x_vec = F.normalize(hand_centroid - obj_centroid, dim=1)  # From object to hand
            if random_rot:
                rand_vec = transforms.random_rotations(hand_verts.shape[0], device=hand_verts.device)   # Generate random rot matrix
                y_vec = F.normalize(torch.cross(x_vec, rand_vec[:, :3, 0]), dim=1)  # Make orthogonal
            else:
                ref_pt = hand_verts[:, 80, :]
                y_vec = F.normalize(torch.cross(x_vec, ref_pt - obj_centroid), dim=1)  # From object to hand ref point

            z_vec = F.normalize(torch.cross(x_vec, y_vec), dim=1)  # Z axis

            tform = ops.eyes(4, hand_verts.shape[0], device=hand_verts.device)
            tform[:, :3, 0] = x_vec
            tform[:, :3, 1] = y_vec
            tform[:, :3, 2] = z_vec
            tform[:, :3, 3] = obj_centroid

            return torch.inverse(tform)

    @staticmethod
    def verts_to_pointcloud(hand_verts, hand_feats, obj_verts, obj_feats):
        """
        Convert hand and object vertices and features from Pytorch3D padded format (batch, vertices, N)
        to Pytorch-Geometric packed format (all_vertices, N)
        :return: x: the features of hand and object (batch*(num_hand_vertices + num_obj_vertices), 25)
        :return: pos: the verts of hand and object (batch*(num_hand_vertices + num_obj_vertices), 3)
        :return: batch: the number of all vertices (batch*(num_hand_vertices + num_obj_vertices),)
        """
        batch_size = hand_verts.shape[0]
        device = hand_verts.device

        ptcloud_pos = torch.cat((hand_verts, obj_verts), dim=1)
        ptcloud_x = torch.cat((hand_feats, obj_feats), dim=1)

        _, N, _ = ptcloud_pos.shape  # (batch_size, num_points, 3)
        pos = ptcloud_pos.view(batch_size * N, -1)
        batch = torch.zeros((batch_size, N), device=device, dtype=torch.long)
        for i in range(batch_size):
            batch[i, :] = i
        batch = batch.view(-1)
        x = ptcloud_x.view(-1, hand_feats.shape[2])

        # print('x', x.shape, pos.shape, batch.shape)
        return x, pos, batch
    
    @staticmethod
    def get_entropy_weight(contact_obj, contact_hand):
        # standardization of the dataset
        """
        According to the difference between positive and negative, standardization of data can be parted into two ways:
        positive: (x-min)/(max-min)
        negative: (max-x)/(max-min)
        """
        contact_obj_max = np.max(contact_obj, axis=2, keepdims=True)
        contact_obj_min = np.min(contact_obj, axis=2, keepdims=True)
        contact_hand_max = np.max(contact_hand, axis=2, keepdims=True)
        contact_hand_min = np.min(contact_hand, axis=2, keepdims=True)

        contact_obj = (contact_obj_max - contact_obj) / (contact_obj_max - contact_obj_min)
        contact_hand = (contact_hand_max - contact_hand) / (contact_hand_max - contact_hand_min)

        # Calculate entropy weight by all batches
        contact_obj = contact_obj.reshape(-1, contact_obj.shape[2])
        contact_hand = contact_hand.reshape(-1,contact_hand.shape[2])

        # calculate entropy and weights
        # compute K
        K_obj = 1/np.log(contact_obj.shape[0])
        K_hand = 1/np.log(contact_hand.shape[0])

        # entropy 
        all_batch_data_obj = np.sum(contact_obj, axis=0)
        all_batch_data_hand = np.sum(contact_hand, axis=0)

        for idx in range(contact_obj.shape[0]): # calculate proportion of obj
            contact_obj[idx] = (contact_obj[idx] + 1e-6) / (all_batch_data_obj + 1e-6) 
            # Adding 1e-6 to the molecule is to prevent when it is 0, ln0=Nan
            # Adding 1e-6 to the denominator is to prevent when denominator is 0, contact_obj = Nan

        for idx in range(contact_hand.shape[0]): # calculate proportion of hand
            contact_hand[idx] = (contact_hand[idx] + 1e-6) / (all_batch_data_hand + 1e-6) 
            # Adding 1e-6 to the molecule is to prevent when it is 0, ln0=Nan
            # Adding 1e-6 to the denominator is to prevent when denominator is 0, contact_obj = Nan

        E_obj = -K_obj * np.sum(contact_obj * np.log(contact_obj), axis=0)
        E_hand = -K_hand * np.sum(contact_hand * np.log(contact_hand), axis=0)

        # w_obj = E_obj / np.sum(E_obj)
        # w_hand = E_hand / np.sum(E_hand)

        D_obj = 1 - E_obj
        D_hand = 1 - E_hand

        w_obj = 10 * D_obj / np.sum(D_obj) # Multiply by 10 to ensure that the loss of the model is not too small, which is convenient for training
        w_hand = 10 * D_hand / np.sum(D_hand)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(w_obj)
        print(w_hand)

        return torch.Tensor(w_obj).to(device), torch.Tensor(w_hand).to(device)