# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from os import path as osp
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import pytorch3d
from manopth import manolayer
import open3d
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
from manopth.manolayer import ManoLayer
import trimesh
import torch.nn.functional as F
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SAMPLE_VERTS_NUM = 2048
DEEPCONTACT_BIN_WEIGHTS_FILE = 'data/class_bin_weights.out'
AFFORDANCE_BIN_WEIGHTS_FILE = 'data/affordance_class_bin_weights.txt'
DEEPCONTACT_NUM_BINS = 10


def val_to_class(val):
    """
    Converts a contact value [0-1] to a class assignment
    :param val: tensor (batch, verts)
    :return: class assignment (batch, verts)
    """
    expanded = torch.floor(val * DEEPCONTACT_NUM_BINS)
    return torch.clamp(expanded, 0, DEEPCONTACT_NUM_BINS - 1).long()    # Cut off potential 1.0 inputs?


def class_to_val(raw_scores):
    """
    Finds the highest softmax for each class
    :param raw_scores: tensor (batch, verts, classes)
    :return: highest class (batch, verts)
    """
    cls = torch.argmax(raw_scores, dim=2)
    val = (cls + 0.5) / DEEPCONTACT_NUM_BINS
    return val


def forward_mano(mano_model, pose, beta, tforms):
    """Forward mano pass, MANO params to mesh"""
    device = pose.device
    batch_size = pose.shape[0]

    verts, joints = mano_model(pose, beta)

    verts_homo = torch.cat((verts / 1000, torch.ones(batch_size, verts.shape[1], 1, device=device)), 2)
    joints_homo = torch.cat((joints / 1000, torch.ones(batch_size, joints.shape[1], 1, device=device)), 2)

    tform_agg = torch.eye(4, device=device).reshape(1, 4, 4).repeat(batch_size, 1, 1)
    for tform in tforms:
        tform_agg = torch.bmm(tform, tform_agg)  # Aggregate all transforms

    # Apply aggregated transform to all points, permuting for matmul
    verts_homo = torch.bmm(tform_agg, verts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    joints_homo = torch.bmm(tform_agg, joints_homo.permute(0, 2, 1)).permute(0, 2, 1)

    return verts_homo[:, :, :3], joints_homo[:, :, :3]


def fit_pca_to_axang(mano_pose, mano_beta):
    """
    This project uses the MANO model parameterized with 15 PCA components. However, many other approaches use
    different parameterizations (15 joints, parameterized with 45 axis-angle parameters). This function
    allows converting between the formats. It first runs the MANO model forwards to get the hand vertices of
    the initial format. Then an optimization is performed to adjust the 15 PCA parameters of a second MANO model
    to match the initial vertices. Perhaps there are better ways to do this, but this ensures highest accuracy.

    :param mano_pose: numpy (45) axis angle coordinates
    :param mano_beta: numpy (10) beta parameters
    :return: numpy (15) PCA parameters of fitted hand

    IDEA: 45 axis angle --> 45 PCA parameters --> 15 PCA parameters(by optimization loss)
    """

    mano_pose = np.array(mano_pose)
    full_axang = torch.Tensor(mano_pose).unsqueeze(0)
    mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=45, side='right', flat_hand_mean=False) # mano layer for intermediate transition variables

    beta_in = torch.Tensor(mano_beta).unsqueeze(0)
    mano_model_orig = ManoLayer(mano_root='mano/models', joint_rot_mode="axisang", use_pca=False, center_idx=None, flat_hand_mean=True) # input hand manolayer
    _, target_joints = forward_mano(mano_model_orig, full_axang, beta_in, []) # obtain target hand joints variable

    full_axang[:, 3:] -= mano_model.th_hands_mean
    pca_mat = mano_model.th_selected_comps.T # PCA matrix
    pca_shape = full_axang[:, 3:].mm(pca_mat)  # Since the HO gt is in full 45 dim axang coords, convert back to PCA shape
    new_pca_shape = np.zeros(18) # take the variables of the first 15 dimensions + 3 dimensions wrist rotation
    new_pca_shape[:3] = mano_pose[:3]   # set axang
    new_pca_shape[3:] = pca_shape[0, :15]   # set pca pose

    # Do optimization
    pca_in = torch.Tensor(new_pca_shape).unsqueeze(0)

    pca_in.requires_grad = True
    mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False) # output mano hand layer
    optimizer = torch.optim.Adam([pca_in], lr=0.03, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss()

    for it in range(200):
        optimizer.zero_grad()
        hand_verts, hand_joints = forward_mano(mano_model, pca_in, beta_in, [])   # 2.2ms
        # vis_pointcloud(hand_joints, target_joints)
        loss = loss_criterion(hand_joints, target_joints)
        # print('Opt loss', loss.detach())
        loss.backward()
        optimizer.step()

    return pca_in.detach().squeeze(0).numpy()

def new_fit_mano_params(mano_pose, mano_beta, input_mano_parm, out_mano_parm):  
    """
    This project uses the MANO model parameterized with 15 PCA components. However, many other approaches use
    different parameterizations (15 joints, parameterized with 45 axis-angle parameters or different PCA parameters of MANO model).
    The original way in ContactOpt has lower performance and can only achieve conversion between 15 PCA components and 45 axis-angle 
    parameters. We improve this way and complete the approximate conversion between different forms of MANO parameters.
    
    It first runs the MANO model forwards to get the hand joints of the initial format. Then an optimization is performed 
    to adjust the out MANO parameters of a second MANO model to match the initial hand joints. 

    :param mano_pose: input MANO hand pose parameters
    :param mano_beta: numpy (10) beta parameters(hand shape parameters)
    :input_mano_parm: input ManoLayer parameters
    :out_mano_parm: out ManoLayer parameters
    :return: numpy optimized Mano hand pose parameters
    """
    input_joint_rot_mode = input_mano_parm['joint_rot_mode']
    input_use_pca = input_mano_parm['use_pca']
    input_flat_hand_mean = input_mano_parm['flat_hand_mean']
    input_ncomps = input_mano_parm['ncomps']
    input_hand_side = input_mano_parm['hand_side']

    output_joint_rot_mode = out_mano_parm['joint_rot_mode']
    output_use_pca = out_mano_parm['use_pca']
    output_flat_hand_mean = out_mano_parm['flat_hand_mean']
    output_ncomps = out_mano_parm['ncomps']
    output_hand_side = out_mano_parm['hand_side']

    mano_pose = np.array(mano_pose)
    full_axang = torch.Tensor(mano_pose).unsqueeze(0)
    mano_model_orig = ManoLayer(mano_root='mano/models', joint_rot_mode=input_joint_rot_mode, use_pca=input_use_pca, center_idx=None, 
                                flat_hand_mean=input_flat_hand_mean, ncomps=input_ncomps, side=input_hand_side)
    beta_in = torch.Tensor(mano_beta).unsqueeze(0)
    target_verts, target_joints = forward_mano(mano_model_orig, full_axang, beta_in, [])

    new_pca_shape = np.zeros(output_ncomps+3) # 3 means hand wrist rotation
    new_pca_shape[:3] = mano_pose[:3]   # set axang

    # Do optimization
    pca_in = torch.Tensor(new_pca_shape).unsqueeze(0)

    pca_in.requires_grad = True
    mano_model = ManoLayer(mano_root='mano/models', joint_rot_mode=output_joint_rot_mode, use_pca=output_use_pca, 
                           ncomps=output_ncomps, side=output_hand_side, flat_hand_mean=output_flat_hand_mean)
    optimizer = torch.optim.Adam([pca_in], lr=0.03, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss()

    for it in range(2000):
        optimizer.zero_grad()
        hand_verts, hand_joints = forward_mano(mano_model, pca_in, beta_in, [])   # 2.2ms
        # vis_pointcloud(hand_joints, target_joints)
        loss = loss_criterion(hand_joints, target_joints)
        # print('Opt loss', loss.detach())
        loss.backward()
        optimizer.step()

    return pca_in.detach().squeeze(0).numpy()


def hand_color():
    return np.asarray([224.0, 172.0, 105.0]) / 255


def obj_color():
    return np.asarray([100.0, 100.0, 100.0]) / 255


def save_trimesh(obj_mesh, output_path):
    obj_raw = trimesh.exchange.obj.export_obj(obj_mesh, include_texture=False)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as obj_file:
        obj_file.write(obj_raw)


def verts_to_name(num_verts):
    """Hacky function allowing finding the name of an object by the number of vertices.
    Each object happens to have a different number."""

    num_verts_dict = {100597: 'mouse', 29537: 'binoculars', 100150: 'bowl', 120611: 'camera', 64874: 'cell_phone',
                      177582: 'cup', 22316: 'eyeglasses', 46334: 'flashlight', 35949: 'hammer', 93324: 'headphones',
                      19962: 'knife', 169964: 'mug', 57938: 'pan', 95822: 'ps_controller', 57824: 'scissors',
                      144605: 'stapler', 19708: 'toothbrush', 42394: 'toothpaste', 126627: 'utah_teapot', 90926: 'water_bottle',
                      104201: 'wine_glass', 108248: 'door_knob', 71188: 'light_bulb', 42232: 'banana', 93361: 'apple',
                      8300: 'HO_sugar', 8251: 'HO_soap', 16763: 'HO_mug', 10983: 'HO_mustard', 9174: 'HO_drill',
                      8291: 'HO_cheezits', 8342: 'HO_spam', 10710: 'HO_banana', 8628: 'HO_scissors',
                      148245: 'train_exclude'}

    if num_verts in num_verts_dict:
        return num_verts_dict[num_verts]

    return 'DIDNT FIND {}'.format(num_verts)


def mesh_is_thin(num_verts):
    """For thin meshes, the interpenetration loss doesn't do anything, since they're thinner than 2*2mm.
    For thin objects, we set this margin to zero mm."""
    thins = [19708, 19962, 22316, 16763, 8628]   # Toothbrush, Knife, Eyeglasses, HO_mug, HO_scissors

    is_thin = torch.zeros_like(num_verts)
    for t in thins:
        is_thin[num_verts == t] = 1

    return is_thin


def upscale_contact(obj_mesh, obj_sampled_idx, contact_obj):
    """
    When we run objects through our network, they always have a fixed number of vertices.
    We need to up/downscale the contact from this to the original number of vertices
    :param obj_mesh: Pytorch3d Meshes object, includes the number of vertices for each mesh in a batch
    :param obj_sampled_idx: (batch, 2048)
    :param contact_obj: (batch, 2048)
    :return:

    idea: 1) padding obj mesh to obtain max obj vertices for covering the all meshes in a batch
    2) based on obj_sample_idx, verify the real sample obj verts. 3) using KNN method, find each vertex corresponds to the index of samping point
    4) based closest_idx, use the contact situation of sampling points to characterize the contact situation of unsampled points.
    """
    obj_verts = obj_mesh.verts_padded() # Get the padded representation of the vertices. Returns: tensor of vertices of shape (N, max(V_n), 3).
    # obj_mesh: pytorch3d.structures.Meshes. 
    # obj_mesh is a list of verts V_n = [[V_1], [V_2], … , [V_N]] where V_1, … , V_N are the number of verts in each mesh and N is the number of meshes.
    # refert to https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/structures/meshes.html
    _, closest_idx, _ = pytorch3d.ops.knn_points(obj_verts, batched_index_select(obj_verts, 1, obj_sampled_idx), K=1) # Find the sampling point index corresponding to each vertex of the object after padding
    upscaled = batched_index_select(contact_obj, 1, closest_idx.squeeze(2)) # based the sampling point index，assign contact values to non-sampled points
    # use the contact situation of sampling points to characterize the contact situation of unsampled points
    return upscaled.detach()


def hack_filedesciptor():
    """
    Sometimes needed if reading datasets very quickly? Fixes:
        RuntimeError: received 0 items of ancdata
    https://github.com/pytorch/pytorch/issues/973
    """
    torch.multiprocessing.set_sharing_strategy('file_system')


def apply_tform(tform, verts):
    """
    Applies a 4x4 rigid transform to a list of points
    :param tform: tensor (batch, 4, 4)
    :param verts: tensor (batch, N, 3)
    :return:
    """
    verts_homo = torch.cat((verts, torch.ones(verts.shape[0], verts.shape[1], 1, device=verts.device)), 2)
    new_verts = torch.bmm(tform, verts_homo.permute(0, 2, 1)).permute(0, 2, 1)
    return new_verts[:, :, :3]


def apply_rot(rot, verts, around_centroid=False):
    """
    Applies a 3x3 rotation matrix to a list of points
    :param rot: tensor (batch, 3, 3)
    :param verts: tensor (batch, N, 3)
    :return:
    """
    if around_centroid: # centroid unchanged
        centroid = verts.mean(dim=1, keepdim=True)
        verts = verts - centroid

    new_verts = torch.bmm(rot, verts.permute(0, 2, 1)).permute(0, 2, 1)

    if around_centroid:
        new_verts = new_verts + centroid

    return new_verts


def translation_to_tform(translation):
    """
    (batch, 3) to (batch, 4, 4)
    """
    tform_out = pytorch3d.ops.eyes(4, translation.shape[0], device=translation.device)
    tform_out[:, :3, 3] = translation
    return tform_out


def sharpen_contact(c, slope=10, thresh=0.6):
    """
    Apply filter to input, makes into a "soft binary"
    """
    out = slope * (c - thresh) + thresh
    return torch.clamp(out, 0.0, 1.0)


def fit_sigmoid(colors, a=0.05):
    """Fits a sigmoid to raw contact temperature readings from the ContactPose dataset. This function is copied from that repo"""
    idx = colors > 0
    ci = colors[idx]

    x1 = min(ci)  # Find two points
    y1 = a
    x2 = max(ci)
    y2 = 1-a

    lna = np.log((1 - y1) / y1)
    lnb = np.log((1 - y2) / y2)
    k = (lnb - lna) / (x1 - x2)
    mu = (x2*lna - x1*lnb) / (lna - lnb)
    ci = np.exp(k * (ci-mu)) / (1 + np.exp(k * (ci-mu)))  # Apply the sigmoid
    colors[idx] = ci
    return colors


def subdivide_verts(edges, verts):
    """
    Takes a list of edges and vertices, and subdivides each edge and puts a vert in the middle. May not work with variable-size meshes
    :param edges: (batch, E, 2)
    :param verts: (batch, V, 3)
    :return: new_verts (batch, E+V, 3)
    """
    selected_verts = edges.view(edges.shape[0], -1)     # Flatten into (batch, E*2)
    new_verts = batched_index_select(verts, 1, selected_verts)
    new_verts = new_verts.view(edges.shape[0], edges.shape[1], 2, 3)
    new_verts = new_verts.mean(dim=2)

    new_verts = torch.cat([verts, new_verts], dim=1)  # (sum(V_n)+sum(E_n), 3)
    return new_verts


def calc_l2_err(a, b, axis=2):
    if torch.is_tensor(a):
        mse = torch.sum(torch.square(a - b), dim=axis)
        l2_err = torch.sqrt(mse)
        return torch.mean(l2_err, 1)
    else:
        mse = np.linalg.norm(a - b, 2, axis=axis)
        return mse.mean()


def batched_index_select(t, dim, inds):
    """
    Helper function to extract batch-varying indicies along array
    :param t: array to select from
    :param dim: dimension to select along
    :param inds: batch-vary indicies
    :return:
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def mesh_set_color(color, mesh, colormap=plt.cm.inferno):
    """
    Applies colormap to object
    :param color: Tensor or numpy array, (N, 1)
    :param mesh: Open3D TriangleMesh
    :return:
    """
    # vertex_colors = np.tile(color.squeeze(), (3, 1)).T
    # mesh.vertex_colors = o3du.Vector3dVector(vertex_colors)
    # geometry.apply_colormap(mesh, apply_sigmoid=False)

    colors = np.asarray(color).squeeze()
    if len(colors.shape) > 1:
        colors = colors[:, 0]

    colors[colors < 0.1] = 0.1 # TODO hack to make brighter

    colors = colormap(colors)[:, :3] # color mapping: Linear color mapping（viridis、plasma、inferno, etc.); Periodic color mapping（hsv、rainbow、jet, etc.)
    colors = o3du.Vector3dVector(colors)
    mesh.vertex_colors = colors


def aggregate_tforms(tforms):
    """Aggregates a list of 4x4 rigid transformation matricies"""
    device = tforms[0].device
    batch_size = tforms[0].shape[0]

    tform_agg = pytorch3d.ops.eyes(4, batch_size, device=device)
    for tform in tforms:
        tform_agg = torch.bmm(tform, tform_agg)  # Aggregate all transforms

    return tform_agg


def axisEqual3D(ax):
    """Sets a matplotlib 3D plot to have equal-scale axes"""
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def vis_pointcloud(object_points, hand_points, idx=None, show=True):
    if show:
        plt.switch_backend('TkAgg')
    else:
        plt.switch_backend('agg')

    if idx is None:
        idx = int(np.random.randint(0, hand_points.shape[0]))   # Select random sample from batch

    object_points = object_points[idx, :, :].detach().cpu().numpy()
    hand_points = hand_points[idx, :, :].detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2])
    ax.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2]) #, c=np.arange(hand_points.shape[0]))

    if show:
        axisEqual3D(ax)
        # plt.axis('off')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    return fig


def get_mano_closed_faces():
    """The default MANO mesh is "open" at the wrist. By adding additional faces, the hand mesh is closed,
    which looks much better.
    https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manoutils.py"""
    mano_layer = manolayer.ManoLayer(
        joint_rot_mode="axisang", use_pca=False, mano_root='mano/models', center_idx=None, flat_hand_mean=True
    )
    close_faces = torch.Tensor(
        [
            [92, 38, 122],
            [234, 92, 122],
            [239, 234, 122],
            [279, 239, 122],
            [215, 279, 122],
            [215, 122, 118],
            [215, 118, 117],
            [215, 117, 119],
            [215, 119, 120],
            [215, 120, 108],
            [215, 108, 79],
            [215, 79, 78],
            [215, 78, 121],
            [214, 215, 121],
        ]
    )
    closed_faces = torch.cat([mano_layer.th_faces, close_faces.long()])
    # Indices of faces added during closing --> should be ignored as they match the wrist
    # part of the hand, which is not an external surface of the human

    # Valid because added closed faces are at the end
    hand_ignore_faces = [1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551]

    return closed_faces.detach().cpu().numpy() #, hand_ignore_faces


def text_3d(text, pos, direction=None, degree=-90.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', font_size=10):
    """
    Generate a Open3D text point cloud used for visualization.
    https://github.com/intel-isl/Open3D/issues/2
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    # font_obj = ImageFont.truetype(font, font_size)
    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = open3d.geometry.PointCloud()
    pcd.colors = open3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    # pcd.points = o3d.utility.Vector3dVector(indices / 100.0)
    pcd.points = open3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def to_cpu_numpy(obj):
    """Convert torch cuda tensors to cpu, numpy tensors"""
    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = to_cpu_numpy(v)
            return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(to_cpu_numpy(v))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def dict_to_device(data, device):
    """Move dict of tensors to device"""
    out = dict()
    for k in data.keys():
        out[k] = data[k].to(device)
    return out

def get_entropy_weight(class_obj_gt, class_hand_gt):
    # standardization of the dataset
    """
    class_obj_gt: (batch_size, feature_dim) —> (32,2048)
    class_hand_gt: (batch_size, feature_dim) —> (32,778)
    According to the difference between positive and negative, standardization of data can be parted into two ways:
    positive: (x-min)/(max-min)
    negative: (max-x)/(max-min)
    """
    num_obj= np.zeros((class_obj_gt.shape[0],10)) # (batch_size, 10)
    num_hand = np.zeros((class_hand_gt.shape[0],10)) # (batch_size, 10)

    for i in range(class_obj_gt.shape[0]):
        for j in range(class_obj_gt.shape[1]):
            class_obj = class_obj_gt[i][j]
            num_obj[i][class_obj] += 1

    for i in range(class_hand_gt.shape[0]):
        for j in range(class_hand_gt.shape[1]):
            class_hand = class_hand_gt[i][j]
            num_hand[i][class_hand] += 1

    # calculate the proportion of each contact category
    num_hand = num_hand / class_hand_gt.shape[1]
    num_obj = num_obj / class_obj_gt.shape[1]
    
    # standardization of the dataset
    """
    num_obj: (N, obj_F_Dim) N denotes object recorded numbers; obj_F_Dim denotes the number of metrics
    num_hand: (N, hand_F_Dim) N denotes hand recorded numbers; hand_F_Dim denotes the number of metrics
    According to the difference between positive and negative, standardization of data can be parted into two ways:
    positive: (x-min)/(max-min)
    negative: (max-x)/(max-min)
    """
    # column: axis=0; row: axis=1
    contact_obj_max = np.max(num_obj, axis=0, keepdims=True)
    contact_obj_min = np.min(num_obj, axis=0, keepdims=True)
    contact_hand_max = np.max(num_hand, axis=0, keepdims=True)
    contact_hand_min = np.min(num_hand, axis=0, keepdims=True)

    # contact_obj = (contact_obj_max - contact_obj) / (contact_obj_max - contact_obj_min)
    # contact_hand = (contact_hand_max - contact_hand) / (contact_hand_max - contact_hand_min)
    num_obj = (num_obj - contact_obj_min) / (contact_obj_max - contact_obj_min)
    num_hand = (num_hand - contact_hand_min) / (contact_hand_max - contact_hand_min)

    # # Calculate entropy weight by all batches
    # contact_obj = contact_obj.reshape(-1, contact_obj.shape[2])
    # contact_hand = contact_hand.reshape(-1,contact_hand.shape[2])

    # calculate entropy and weights
    # compute K
    K_obj = 1 / np.log(num_obj.shape[0])
    K_hand = 1 / np.log(num_hand.shape[0])

    # entropy 
    all_batch_data_obj = np.sum(num_obj, axis=0)
    all_batch_data_hand = np.sum(num_hand, axis=0)

    for idx in range(num_obj.shape[0]): # calculate proportion of obj
        num_obj[idx] = (num_obj[idx] + 1e-6) / (all_batch_data_obj + 1e-6) 
        # Adding 1e-6 to the molecule is to prevent when it is 0, ln0=Nan
        # Adding 1e-6 to the denominator is to prevent when denominator is 0, contact_obj = Nan

    for idx in range(num_hand.shape[0]): # calculate proportion of hand
        num_hand[idx] = (num_hand[idx] + 1e-6) / (all_batch_data_hand + 1e-6) 
        # Adding 1e-6 to the molecule is to prevent when it is 0, ln0=Nan
        # Adding 1e-6 to the denominator is to prevent when denominator is 0, contact_obj = Nan

    E_obj = -K_obj * np.sum(num_obj * np.log(num_obj), axis=0)
    E_hand = -K_hand * np.sum(num_hand * np.log(num_hand), axis=0)

    # w_obj = E_obj / np.sum(E_obj)
    # w_hand = E_hand / np.sum(E_hand)

    D_obj = 1 - E_obj
    D_hand = 1 - E_hand

    w_obj = 1 + D_obj / np.sum(D_obj) # Multiply by 10 to ensure that the loss of the model is not too small, which is convenient for training
    w_hand = 1 + D_hand / np.sum(D_hand)
    # w_obj[0] = 1 # No treatment for contact level 0
    # w_hand[1] = 1 # No treatment for contact level 0
    # w_obj = D_obj / np.sum(D_obj) # Multiply by 10 to ensure that the loss of the model is not too small, which is convenient for training
    # w_hand = D_hand / np.sum(D_hand)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # w_obj = F.sigmoid(torch.Tensor(w_obj).to(device))
    # w_hand = F.sigmoid(torch.Tensor(w_hand).to(device))

    return torch.Tensor(w_obj).to(device), torch.Tensor(w_hand).to(device)
    # return 1+w_obj, 1+w_hand

def get_gt_distribution(class_hand_gt, class_obj_gt):
    num_obj_gt= torch.zeros((class_obj_gt.shape[0],10))
    num_hand_gt = torch.zeros((class_hand_gt.shape[0],10))

    for i in range(class_obj_gt.shape[0]):
        for j in range(class_obj_gt.shape[1]):
            obj_gt = class_obj_gt[i][j]
            num_obj_gt[i][obj_gt.type(torch.long)] += 1

    for i in range(class_hand_gt.shape[0]):
        for j in range(class_hand_gt.shape[1]):
            hand_gt = class_hand_gt[i][j]
            num_hand_gt[i][hand_gt.type(torch.long)] += 1

    obj_gt_freq = num_obj_gt / torch.sum(num_obj_gt, axis=1, keepdims=True) # (10,)
    hand_gt_freq = num_hand_gt / torch.sum(num_hand_gt, axis=1, keepdims=True) # (10,)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return obj_gt_freq.to(device), hand_gt_freq.to(device)

# def get_distribution_feature(data):
#     data = F.softmax(data, dim=-1)
#     data = data.view(data.shape[0], -1)    
    
#     lin1 = torch.nn.Linear(data.shape[1], int(data.shape[1]/2))
#     global_lin = torch.nn.Linear(int(data.shape[1]/2), 10)

#     x = lin1(data) # torch.nn.Linear
#     x = global_lin(x)
    
#     return F.sigmoid(x)

def load_contact_zones(file_path='./data/contact_zones.pkl', sort=True):
    with open(file_path, "rb") as f:
        contact_data = pickle.load(f)
        zones = contact_data['contact_zones']

        f.close()
    
    if sort:
        merged_dict = np.concatenate([zones[key] for key in zones.keys()])
        contact_hand_pd = np.sort(merged_dict)
    else:
        contact_hand_pd = zones

    return contact_hand_pd

def display_contact_zones():
    batch_size = 1

    ncomps = 15

    contact_data = load_contact_zones(sort=False)

    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps= ncomps)

    flat_shape = torch.zeros(batch_size, 10)
    flat_pose = torch.zeros(batch_size, ncomps+3)

    hand_verts, hand_joints = mano_layer(flat_pose, flat_shape)

    display_hand({"verts": hand_verts, 'joints':hand_joints}, mano_faces=mano_layer.th_faces, contact_pd_id=contact_data)

def display_contact_zones_o3d():
    batch_size = 1

    ncomps = 15

    contact_data = load_contact_zones(sort=False)

    mano_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps= ncomps)

    flat_shape = torch.zeros(batch_size, 10)
    flat_pose = torch.zeros(batch_size, ncomps+3)

    hand_verts, hand_joints = mano_layer(flat_pose, flat_shape)

    display_hand_v2({"verts": hand_verts, 'joints':hand_joints}, mano_faces=mano_layer.th_faces, contact_data=contact_data)

def display_hand(hand_info, mano_faces=None, contact_pd_id=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    if contact_pd_id is None:
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    else:
        color = ['dark', 'orange', 'red', 'green', 'blue', 'yellow']
        for color_idx, contact_key in enumerate(contact_pd_id.keys()):
            if color_idx == 0:
                continue
            ax.scatter(verts[contact_pd_id[contact_key], 0], verts[contact_pd_id[contact_key], 1], verts[contact_pd_id[contact_key], 2], color=color[color_idx])
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()

def display_hand_v2(hand_info, mano_faces, contact_data):
    hand_verts = hand_info['verts'][0].detach().clone().numpy() #(778,3)
    mano_faces = mano_faces.detach().clone().numpy() #(1538,3)

    vis_mesh = open3d.geometry.TriangleMesh()
    vis_mesh.vertices = open3d.utility.Vector3dVector(hand_verts)
    vis_mesh.triangles = open3d.utility.Vector3iVector(mano_faces)

    verts_color = np.ones_like(hand_verts) * 0.86 # default color: gray
    color = np.array([
        np.array([255.0, 165.0, 0.0])/255.0,
        [1.0,0.0,0.0],
        [0.0,1.0,0.0],
        [0.0,0.0,1.0],
        [1.0,1.0,0.0]
    ])

    for color_idx, contact_key in enumerate(contact_data.keys()):
        if color_idx == 0:
            continue
        verts_color[contact_data[contact_key]] = color[color_idx-1]
    
    vis_mesh.vertex_colors = open3d.utility.Vector3dVector(verts_color)

    # open3d.visualization.draw_geometries([vis_mesh])
    
    # save visualization
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(vis_mesh)

    view_ctl = vis.get_view_control()
    view_ctl.set_up((-0.9955335768044975, 0.052251647303671545, -0.078629910396086827))
    view_ctl.set_front((-0.060608914527854774, -0.99230644576567262, 0.10795590382954305))
    view_ctl.set_lookat((17.518829345703125, 3.9039039611816406, 11.277820587158203))
    view_ctl.set_zoom(0.69999999999999996)

    #render image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("mesh_render.png")
    vis.destroy_window()

    image = Image.open("mesh_render.png")
    pdf_path = "mesh_render.pdf"
    image.save(pdf_path, "PDF", resolution=100.0)

    print(f"渲染结果已保存为: {pdf_path}")

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

def calc_std(data, align_magn=True):
    if align_magn:
        data = 0.1 * data.std() # *0.1 is deal with order-of-magnitude inconsistencies caused by not closing the mesh
    else:
        data = data.std()

    return data

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)