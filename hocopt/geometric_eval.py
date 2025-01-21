# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import hand_object
import os
import util
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation as R
import trimesh
from open3d import io as o3dio
from open3d import geometry as o3dg
from open3d import utility as o3du
from open3d import visualization as o3dv
import matplotlib.pyplot as plt
import torch
import trimesh
import open3d as o3d
import copy


def np_apply_tform(points, tform):
    """
    The non-batched numpy version
    :param points: (N, 3)
    :param tform: (4, 4)
    :return:
    """
    points_homo = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points_out = np.matmul(tform, points_homo.T).T
    return points_out[:, :3]


def get_hand_align_tform(hand_joints):
    """
    Find a 4x4 rigid transform to align the joints of a hand to a 'cardinal rotation'
    :param hand_joints: (21, 3)
    :return: tform: (4, 4)
    """
    center_joint = 0
    x_joint = 2
    y_joint = 17

    trans = hand_joints[center_joint, :]

    x_vec = hand_joints[x_joint, :] - hand_joints[center_joint, :] # thumb vector
    x_vec = x_vec / np.linalg.norm(x_vec)

    y_vec = hand_joints[y_joint, :] - hand_joints[center_joint, :] # little finger vector
    y_vec = np.cross(x_vec, y_vec)
    y_vec = y_vec / np.linalg.norm(y_vec)

    z_vec = np.cross(x_vec, y_vec)
    z_vec = z_vec / np.linalg.norm(z_vec)

    tform = np.eye(4)
    tform[:3, 0] = x_vec
    tform[:3, 1] = y_vec
    tform[:3, 2] = z_vec
    tform[:3, 3] = trans
    # tform: mano2camera

    return np.linalg.inv(tform) # camera2mano


def calc_procrustes(points1, points2, return_tform=False):
    """ Align the predicted entity in some optimality sense with the ground truth.
    Does NOT align scale
    https://github.com/shreyashampali/ho3d/blob/master/eval.py """

    t1 = points1.mean(0)    # Find centroid
    t2 = points2.mean(0)
    points1_t = points1 - t1   # Zero mean
    points2_t = points2 - t2

    R, s = orthogonal_procrustes(points1_t, points2_t)    # Run procrustes alignment, returns rotation matrix and scale
    # R: find an orthogonal matrix R that most closely maps A to B

    points2_t = np.dot(points2_t, R.T)  # Apply tform to second pointcloud
    points2_t = points2_t + t1 # align the points1

    if return_tform:
        return R, t1 - t2
    else:
        return points2_t


def align_by_tform(mtx, tform):
    t2 = mtx.mean(0)
    mtx_t = mtx - t2
    R, t1 = tform
    return np.dot(mtx_t, R.T) + t1 + t2


def get_trans_rot_err(points1, points2):
    """
    Given two pointclouds, find the error in centroid and rotation
    :param points1: numpy (V, 3)
    :param points2: numpy (V, 3)
    :return: translation error (meters), rotation error (degrees)
    """
    tform = calc_procrustes(points1, points2, return_tform=True) # [R,t]: R:(3,3); t:(3,)
    translation_error = np.linalg.norm(tform[1], 2)
    r = R.from_matrix(tform[0])
    rotation_error = r.magnitude() * 180 / np.pi # radians --> angles

    return translation_error, rotation_error

def get_ADD_scores():
    pass

def get_Intersection_Volume(ho_test):
    hand_mesh, obj_mesh = ho_test.get_o3d_meshes(hand_contact=False, normalize_pos=True)

    hand_verts = np.array(hand_mesh.vertices)
    hand_faces = np.array(hand_mesh.triangles)
    hand_trimesh = trimesh.Trimesh(vertices=hand_verts, faces=hand_faces)

    obj_verts = np.array(obj_mesh.vertices)
    obj_faces = np.array(obj_mesh.triangles)
    obj_trimesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
    fill_complete = obj_trimesh.fill_holes()
    # print(fill_complete)
    # if not obj_trimesh.is_watertight:
    #     print("wrong mesh")

    try:
        intersection = trimesh.boolean.intersection([hand_trimesh, obj_trimesh], engine='blender')
    except:
        # Note: Some object meshes are not closed leading to trimesh error. Ball pivoting approach is used for reconstructing the object surface.
        obj_pd = obj_mesh.sample_points_poisson_disk(util.SAMPLE_VERTS_NUM + 4000)
        # obj_rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(obj_pd, 0.01)
        obj_rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(obj_pd, o3d.utility.DoubleVector([0.006, 0.012, 0.03, 0.04]))
        obj_rec_vertices = np.array(obj_rec_mesh.vertices)
        obj_rec_faces = np.array(obj_rec_mesh.triangles)
        obj_trimesh = trimesh.Trimesh(vertices=obj_rec_vertices, faces=obj_rec_faces)
        intersection = trimesh.boolean.intersection([hand_trimesh, obj_trimesh], engine='blender')

    # print(intersection.volume)

    return intersection.volume

def get_Contact_IoU(hand_contact, obj_contact, hand_contact_gt, obj_contact_gt):
    hand_contact_class = util.val_to_class(torch.Tensor(hand_contact)).clone().detach().cpu().numpy()
    obj_contact_class = util.val_to_class(torch.Tensor(obj_contact)).clone().detach().cpu().numpy()
    hand_contact_gt_class = util.val_to_class(torch.Tensor(hand_contact_gt)).clone().detach().cpu().numpy()
    obj_contact_gt_class = util.val_to_class(torch.Tensor(obj_contact_gt)).clone().detach().cpu().numpy()

    hand_contact_zone = np.zeros_like(hand_contact_class)
    hand_contact_zone[hand_contact_class > 0] = 1
    obj_contact_zone = np.zeros_like(obj_contact_class)
    obj_contact_zone[obj_contact_class > 0] = 1
    hand_contact_gt_zone = np.zeros_like(hand_contact_gt_class)
    hand_contact_gt_zone[hand_contact_gt_class > 0] = 1
    obj_contact_gt_zone = np.zeros_like(obj_contact_gt_class)
    obj_contact_gt_zone[obj_contact_gt_class > 0] = 1

    contact_zone = np.concatenate((hand_contact_zone, obj_contact_zone), axis=0)
    contact_zone_gt = np.concatenate((hand_contact_gt_zone, obj_contact_gt_zone), axis=0)

    contact_IoU = np.logical_and(contact_zone, contact_zone_gt).sum() / np.logical_or(contact_zone, contact_zone_gt).sum()

    return contact_IoU

def get_F_score(ho_test, ho_gt, th):
    hand_mesh, obj_mesh = ho_test.get_o3d_meshes()
    hand_mesh_gt, obj_mesh_gt = ho_gt.get_o3d_meshes()

    inter_mesh = hand_mesh + obj_mesh
    # inter_mesh = hand_mesh
    inter_pd = o3d.geometry.PointCloud()
    inter_pd.points = inter_mesh.vertices
    
    inter_mesh_gt = hand_mesh_gt + obj_mesh_gt
    # inter_mesh_gt = hand_mesh_gt 
    inter_pd_gt = o3d.geometry.PointCloud()
    inter_pd_gt.points = inter_mesh_gt.vertices

    d1 = inter_pd.compute_point_cloud_distance(inter_pd_gt) # pre -> gt
    d2 = inter_pd_gt.compute_point_cloud_distance(inter_pd) # gt -> pre
    precision = float(sum(0.000001 <= d < th for d in d1)) / float(len(d1) - len(obj_mesh.vertices))
    recall = float(sum(0.000001 <= d < th for d in d2)) / float(len(d2) - len(obj_mesh_gt.vertices))
    # precision = float(sum(0 <= d < th for d in d1)) / float(len(d1))
    # recall = float(sum(0 <= d < th for d in d2)) / float(len(d2))

    f_score = 2 * recall * precision / (recall + precision + 0.000001)

    return f_score

def geometric_eval(ho_test, ho_gt):
    """
    Computes many statistics about ground truth and HO

    Note that official HO-3D metrics are available here, but they only consider the hand, and I think they do too much alignment
    https://github.com/shreyashampali/ho3d/blob/master/eval.py

    :param ho_test: hand-object under test
    :param ho_gt: ground-truth hand-object
    :return: dictionary of stats
    """
    stats = dict()
    stats['unalign_hand_verts'] = util.calc_l2_err(ho_gt.hand_verts, ho_test.hand_verts, axis=1) # MPVPE
    stats['unalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints, ho_test.hand_joints, axis=1) # MPJPE
    stats['unalign_obj_verts'] = util.calc_l2_err(ho_gt.obj_verts, ho_test.obj_verts, axis=1) # MPVPE
    # stats['ADD_obj_verts'] = 

    calculate_IV = False
    if calculate_IV: # compute this metric cost too much time, generally set calculate_IV False.
        IV_value = get_Intersection_Volume(ho_test)
        stats['intersection_volume_ho'] = IV_value

    max_pene_depth = ho_test.calc_penetration_depth()
    stats['max_penetration_depth'] = max_pene_depth

    # contact_IoU = get_Contact_IoU(ho_test.hand_contact, ho_test.obj_contact, ho_gt.hand_contact, ho_gt.obj_contact)
    # stats['Contact_IoU'] = contact_IoU

    f_score_5 = get_F_score(copy.deepcopy(ho_test), copy.deepcopy(ho_gt), th=0.005)
    f_score_10 = get_F_score(copy.deepcopy(ho_test), copy.deepcopy(ho_gt), th=0.01)
    stats['F@5mm'] = f_score_5
    stats['F@10mm'] = f_score_10

    ############
    t_err, r_err = get_trans_rot_err(ho_gt.obj_verts, ho_test.obj_verts)
    #################

    # align hand root
    root_test = ho_test.hand_joints[0, :] # ho_test.hand_joints (21,3)
    root_gt = ho_gt.hand_joints[0, :]

    stats['rootalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints - root_gt, ho_test.hand_joints - root_test, axis=1)
    stats['rootalign_obj_verts'] = util.calc_l2_err(ho_gt.obj_verts - root_gt, ho_test.obj_verts - root_test, axis=1)

    # align obj center
    obj_cent_gt = ho_gt.obj_verts.mean(0)
    obj_cent_test = ho_test.obj_verts.mean(0)
    stats['objalign_hand_joints'] = util.calc_l2_err(ho_gt.hand_joints - obj_cent_gt, ho_test.hand_joints - obj_cent_test, axis=1)

    # Procrustes analysis, ignoring the translation and rotation between reconstruction and gt
    hand_joints_align_gt = np_apply_tform(ho_gt.hand_joints, get_hand_align_tform(ho_gt.hand_joints))
    hand_joints_align_test = np_apply_tform(ho_test.hand_joints, get_hand_align_tform(ho_test.hand_joints))
    hand_verts_align_gt = np_apply_tform(ho_gt.hand_verts, get_hand_align_tform(ho_gt.hand_joints))
    hand_verts_align_test = np_apply_tform(ho_test.hand_verts, get_hand_align_tform(ho_test.hand_joints))

    stats['handalign_hand_joints'] = util.calc_l2_err(hand_joints_align_gt, hand_joints_align_test, axis=1) # PA-MPJPE
    stats['handalign_hand_verts'] = util.calc_l2_err(hand_verts_align_gt, hand_verts_align_test, axis=1) # PA-MPVPE

    stats['verts'] = ho_gt.obj_verts.shape[0] # number of the obj verts

    return stats

