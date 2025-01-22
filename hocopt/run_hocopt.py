# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys 
sys.path.append("/home/barry/cxg/HOCopt/")

from hocopt.loader import ContactDBDataset
from hocopt.cren_net import CREN
import glob
import argparse
from hocopt.optimize_pose import optimize_pose
from hocopt.visualize import show_optimization
import pickle
from hocopt.hand_object import HandObject
import hocopt.util as util
from tqdm import tqdm
import hocopt.arguments as arguments
import time
import torch
import os
from torch.utils.data import DataLoader
import pytorch3d
import numpy as np
import torch.nn.functional as F


def get_newest_checkpoint():
    """
    Finds the newest model checkpoint file, sorted by the date of the file
    :return: Model with loaded weights
    """
    list_of_files = glob.glob('checkpoints/*.pt')
    latest_file = max(list_of_files, key=os.path.getctime)
    print('Loading checkpoint file:', latest_file)

    model = CREN(test=True)
    pretrained_dict=torch.load(latest_file)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'pointnet' in k}
    model.load_state_dict(pretrained_dict)
    return model

def get_checkpoint(checkpoint):
    print('Loading checkpoint file:', checkpoint)
    model = CREN(test=True)
    pretrained_dict=torch.load(checkpoint)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'pointnet' in k}
    model.load_state_dict(pretrained_dict)
    return model


def run_contactopt(args):
    """
    Actually run ContactOpt approach. Estimates target contact with DeepContact,
    then optimizes it. Performs random restarts if selected. 
    Saves results to a pkl file.
    :param args: input settings
    """
    print('Running split', args.split)
    dataset = ContactDBDataset(args.test_dataset, min_num_cont=args.min_cont)
    
    shuffle = args.vis or args.partial > 0
    print('Shuffle:', shuffle)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=6, collate_fn=ContactDBDataset.collate_fn)
    # shuffle: whether to disrupt the order of load data; if true, the unordered data; else the order data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint != '':
        model = get_checkpoint(args.checkpoint)
    else:
        model = get_newest_checkpoint()
    model.to(device)
    model.eval()

    all_data = list()

    for idx, data in enumerate(tqdm(test_loader)):
        data_gpu = util.dict_to_device(data, device)
        batch_size = data['obj_sampled_idx'].shape[0]

        if args.split != 'fine':
            with torch.no_grad():
                network_out = model(data_gpu['hand_verts_aug'], data_gpu['hand_feats_aug'], data_gpu['obj_sampled_verts_aug'], data_gpu['obj_feats_aug'])
                network_out['contact_obj'] = F.log_softmax(network_out['contact_obj'], dim=-1) # turn into the original network output format
                network_out['contact_hand'] = F.log_softmax(network_out['contact_hand'], dim=-1) # turn into the original network output format
                hand_contact_target = util.class_to_val(network_out['contact_hand']).unsqueeze(2)
                obj_contact_target = util.class_to_val(network_out['contact_obj']).unsqueeze(2)
        else:
            hand_contact_target = data_gpu['hand_contact_gt']
            obj_contact_target = util.batched_index_select(data_gpu['obj_contact_gt'], 1, data_gpu['obj_sampled_idx'])

        if args.sharpen_thresh > 0: # If flag, sharpen contact, (outstanding contact value, for smaller contact values)
            print('Sharpening')
            obj_contact_target = util.sharpen_contact(obj_contact_target, slope=2, thresh=args.sharpen_thresh)
            hand_contact_target = util.sharpen_contact(hand_contact_target, slope=2, thresh=args.sharpen_thresh)

        if args.rand_re > 1:    # If we desire random restarts (different views of camera observe the mano)
            mtc_orig = data_gpu['hand_mTc_aug'].detach().clone()
            print('Doing random optimization restarts')
            best_loss = torch.ones(batch_size) * 100000

            for re_it in range(args.rand_re): 
                # add noise to hand translation and rotation for initializing optimized pose to avoid optimization falling into local  minima
                data_gpu['hand_mTc_aug'] = mtc_orig.detach().clone()
                random_rot_mat = pytorch3d.transforms.euler_angles_to_matrix(torch.randn((batch_size, 3), device=device) * args.rand_re_rot / 180 * np.pi, 'ZYX')
                # pytorch3d.transforms.euler_angles_to_matrix: convert rotations given as Euler angles in radians(Rad) to rotation matrices.
                data_gpu['hand_mTc_aug'][:, :3, :3] = torch.bmm(random_rot_mat, data_gpu['hand_mTc_aug'][:, :3, :3])
                data_gpu['hand_mTc_aug'][:, :3, 3] += torch.randn((batch_size, 3), device=device) * args.rand_re_trans

                cur_result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                           w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                           w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                           w_opt_rot=args.w_opt_rot,
                                           caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                           caps_on_hand=args.caps_hand,
                                           contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                           w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)
                if re_it == 0:
                    out_pose = torch.zeros_like(cur_result[0])
                    out_mTc = torch.zeros_like(cur_result[1])
                    obj_rot = torch.zeros_like(cur_result[2])
                    opt_state = cur_result[3]

                loss_val = cur_result[3][-1]['loss']
                for b in range(batch_size):
                    if loss_val[b] < best_loss[b]:
                        best_loss[b] = loss_val[b]
                        out_pose[b, :] = cur_result[0][b, :]
                        out_mTc[b, :, :] = cur_result[1][b, :, :]
                        obj_rot[b, :, :] = cur_result[2][b, :, :]

                # print('Loss, re', re_it, loss_val)
                # print('Best loss', best_loss)
        else:
            result = optimize_pose(data_gpu, hand_contact_target, obj_contact_target, n_iter=args.n_iter, lr=args.lr,
                                   w_cont_hand=args.w_cont_hand, w_cont_obj=1, save_history=args.vis, ncomps=args.ncomps,
                                   w_cont_asym=args.w_cont_asym, w_opt_trans=args.w_opt_trans, w_opt_pose=args.w_opt_pose,
                                   w_opt_rot=args.w_opt_rot,
                                   caps_top=args.caps_top, caps_bot=args.caps_bot, caps_rad=args.caps_rad,
                                   caps_on_hand=args.caps_hand,
                                   contact_norm_method=args.cont_method, w_pen_cost=args.w_pen_cost,
                                   w_obj_rot=args.w_obj_rot, pen_it=args.pen_it)
            out_pose, out_mTc, obj_rot, opt_state = result

        obj_contact_upscale = util.upscale_contact(data_gpu['mesh_aug'], data_gpu['obj_sampled_idx'], obj_contact_target)
        # obj_mesh + obj_sample_idx + obj_contact_target --> obj_full_contact for all vertices

        for b in range(obj_contact_upscale.shape[0]):    # Loop over batch
            gt_ho = HandObject()
            in_ho = HandObject()
            out_ho = HandObject()
            gt_ho.load_from_batch(data['hand_beta_gt'], data['hand_pose_gt'], data['hand_mTc_gt'], data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_gt'], b)
            in_ho.load_from_batch(data['hand_beta_aug'], data['hand_pose_aug'], data['hand_mTc_aug'], hand_contact_target, obj_contact_upscale, data['mesh_aug'], b)
            out_ho.load_from_batch(data['hand_beta_aug'], out_pose, out_mTc, data['hand_contact_gt'], data['obj_contact_gt'], data['mesh_aug'], b, obj_rot=obj_rot)
            # out_ho.calc_dist_contact(hand=True, obj=True)
            # .load_from_batch() cut off padding parts of object contact information
            # .load_from_batch() saves hand_beta、hand_pose、hand_mTc、hand_contact、obj_verts、obj_faces、obj_contact

            all_data.append({'gt_ho': gt_ho, 'in_ho': in_ho, 'out_ho': out_ho})

        if args.vis:
            show_optimization(data, opt_state, hand_contact_target.detach().cpu().numpy(), obj_contact_upscale.detach().cpu().numpy(),
                              is_video=args.video, vis_method=args.vis_method)

        if idx >= args.partial >= 0:   # Speed up for eval i.e. idx >= args.partial && args.partial > 0
            break

    #------- save optimized result -------------# 
    # out_file = 'data/optimized_{}.pkl'.format(args.split)
    out_file = 'data/optimized_test.pkl'
    print('Saving to {}. Len {}'.format(out_file, len(all_data)))
    if args.compressed_storage:
        with open(out_file, 'wb') as output_file:
            seq_length = int(len(all_data) / args.subfile_num)
            rest_length = len(all_data) - args.subfile_num*seq_length
            rest_flag = dict()
            rest_flag.update({'rest_flag': True}) if rest_length > 0 else rest_flag.update({'rest_flag': False})
            pickle.dump(rest_flag, output_file)
            for _ in range(args.subfile_num):
                pickle.dump(all_data[:seq_length], output_file)
                del all_data[:seq_length]
            if rest_length > 0:
                pickle.dump(all_data[:rest_length], output_file)
                del all_data[:rest_length]
            del all_data
            output_file.close()
    else:
        pickle.dump(all_data, open(out_file, 'wb'))


if __name__ == '__main__':
    util.hack_filedesciptor() # ensure load dataset quickly
    args = arguments.run_contactopt_parse_args()

    if args.split == 'aug' or args.split == 'handoff':     # Settings defaults for Perturbed ContactPose
        defaults = {'lr': 0.005,
                    'n_iter': 300,
                    'w_cont_hand': 2.0, # 2.0
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 3,
                    'w_opt_trans': 1.3, # ## 1.3
                    'w_opt_rot': 0.9, # ## 1.0 0.9
                    'w_opt_pose': 0.9, ##1.0  0.9 not use 1.2
                    'caps_rad': 0.001,
                    'cont_method': 6, ## 0
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 700, # 600 700
                    'pen_it': 0,
                    'rand_re': 8, #
                    'rand_re_trans': 0.05, # ## 0.04
                    'rand_re_rot': 6, ## 5
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'im' or args.split == 'demo':    # Settings defaults for image-based pose estimates
        defaults = {'lr': 0.01,
                    'n_iter': 250,
                    'w_cont_hand': 2.5,
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 2,
                    'w_opt_trans': 0.3,
                    'w_opt_rot': 1,
                    'w_opt_pose': 1.0,
                    'caps_rad': 0.001,
                    'cont_method': 6,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 320,
                    'pen_it': 0,
                    'rand_re': 8,
                    'rand_re_trans': 0.02,
                    'rand_re_rot': 5,
                    'w_obj_rot': 0,
                    'vis_method': 1}
    elif args.split == 'im_dexycb':    # Settings defaults for image-based pose estimates(DexYCB/HO3D)
        # need to adjust function "run_mano" in hand_object.py --> mano_model = ManoLayer(mano_root='mano/models', joint_rot_mode="axisang", use_pca=True, ncomps=45, center_idx=None, flat_hand_mean=False)
        defaults = {'lr': 0.001,
                    'n_iter': 100,
                    'w_cont_hand': 3.5, # 2.0
                    'sharpen_thresh': 0.1,
                    'ncomps': 15,
                    'w_cont_asym': 3.5, # 2.0
                    'w_opt_trans': 0.003, # 0.03
                    'w_opt_rot': 0.9,
                    'w_opt_pose': 3.0, # 0.9
                    'caps_rad': 0.001,
                    'cont_method': 6,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 1200,
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.0,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}
    elif args.split == 'im_ho3d':    # Settings defaults for image-based pose estimates(DexYCB/HO3D)
        # need to adjust function "run_mano" in hand_object.py --> mano_model = ManoLayer(mano_root='mano/models', joint_rot_mode="axisang", use_pca=False, ncomps=45, center_idx=None, flat_hand_mean=True)
        defaults = {'lr': 0.001,
                    'n_iter': 300,
                    'w_cont_hand': 1.0, # 1.0
                    'sharpen_thresh': -1,
                    'ncomps': 15,
                    'w_cont_asym': 3.0, # 3
                    'w_opt_trans': 0.03,
                    'w_opt_rot': 0.9,
                    'w_opt_pose': 1.5, # 1.5
                    'caps_rad': 0.001,
                    'cont_method': 6,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 900, # 900
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.0,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}
    elif args.split == 'fine':  # Settings defaults for small-scale refinement
        defaults = {'lr': 0.003,
                    'n_iter': 250,
                    'w_cont_hand': 3.0, ##
                    'sharpen_thresh': 0.1,
                    'ncomps': 15,
                    'w_cont_asym': 4,
                    'w_opt_trans': 0.03,
                    'w_opt_rot': 0.9,
                    'w_opt_pose': 0.9,
                    'caps_rad': 0.001,
                    'cont_method': 6,
                    'caps_top': 0.0005,
                    'caps_bot': -0.001,
                    'w_pen_cost': 600,
                    'pen_it': 0,
                    'rand_re': 1,
                    'rand_re_trans': 0.00,
                    'rand_re_rot': 0,
                    'w_obj_rot': 0,
                    'vis_method': 5}

    for k in defaults.keys():   # Override arguments that have not been manually set with defaults
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]
        # vars(): return a dictionary object of the attributes and attribute values {"attribute1":value1, "attribute2":value2, "attribute3":value3, ...}

    print(args)

    start_time = time.time()
    run_contactopt(args)
    print('Elapsed time:', time.time() - start_time)

