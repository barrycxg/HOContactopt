# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys 
sys.path.append("/home/barry/cxg/HOCopt/")

import pickle
from open3d import visualization as o3dv
import random
import argparse
import numpy as np
import time
import hocopt.util as util
import hocopt.geometric_eval as geometric_eval
import pprint
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn.metrics
import trimesh
import torch
import copy
import os
import open3d
import cv2
import time

SAVE_OBJ_FOLDER = 'eval/saveobj'

def video_make(img_dir_path, idx):
    img_array = []

    for img in sorted(os.listdir(img_dir_path), key = lambda x: int(x.split('.')[0])):
        img_path = os.path.join(img_dir_path, img)
        img = cv2.imread(img_path)
        if img is None:
            print(img_path + " is error!")
            continue

        img_array.append(img)

    size = img_array[0].shape[:2] # (width, length) ——> videowriter (length, width)

    save_demo_path = 'exp/demo_video_' + args.split
    os.makedirs(save_demo_path, exist_ok=True)
    videowrite = cv2.VideoWriter(os.path.join(save_demo_path, 'demo_%d_HFLNet.mp4'%idx), 0x7634706d, 20.0, (size[1], size[0])) # 0x7634706d mp4v, -1 view all supported formats
    for i in range(len(os.listdir(img_dir_path))):
        videowrite.write(img_array[i])
    videowrite.release()

def plot_3d(hand_gt, obj_gt, hand_out, obj_out, hand_in, obj_in, idx, save_video=False, save_path_video='exp/demo_img_'):
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    # vis.add_geometry(hand_gt)
    # vis.add_geometry(obj_gt)
    # vis.add_geometry(hand_out)
    # vis.add_geometry(obj_out)
    vis.add_geometry(hand_in)
    vis.add_geometry(obj_in)
    # vis.add_geometry(util.text_3d('In', pos=[-0.4, 0.2, 0], font_size=40, density=2))
    # vis.add_geometry(util.text_3d('Refined', pos=[-0.4, 0.4, 0], font_size=40, density=2))
    # vis.add_geometry(util.text_3d('GT', pos=[-0.4, 0.0, 0], font_size=40, density=2))

    save_path_video = save_path_video + args.split
    save_path_video = os.path.join(save_path_video, "%d"%idx)
    os.makedirs(save_path_video, exist_ok=True)
    
    rotation_angle = 1
    num_rotate = 0

    while True:
        ## Rotate the mesh at each iteration
        # R1 = np.asarray(vis_hand.get_rotation_matrix_from_axis_angle([0, np.radians(rotation_angle), 0]))
        # R2 = np.asarray(vis_obj.get_rotation_matrix_from_axis_angle([0, np.radians(rotation_angle), 0]))
        R = np.asarray(open3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(rotation_angle), 0])) # rotation angle is fixed, due to the need for uniform rotation
        # generate rotation matrix from axis_angle
        # hand_gt.rotate(R, center=hand_gt.get_center())
        # obj_gt.rotate(R, center=hand_gt.get_center())
        # hand_out.rotate(R, center=hand_out.get_center())
        # obj_out.rotate(R, center=hand_out.get_center())
        hand_in.rotate(R, center=hand_in.get_center())
        obj_in.rotate(R, center=hand_in.get_center())
        # hand and obj should be mounted with the fixed center(such as hand_center or [0,0,0]) rotation

        # update geometry
        # vis.update_geometry(hand_gt)
        # vis.update_geometry(obj_gt)
        # vis.update_geometry(hand_out)
        # vis.update_geometry(obj_out)
        vis.update_geometry(hand_in)
        vis.update_geometry(obj_in)
        vis.poll_events()
        vis.update_renderer()

        if num_rotate==0:
            time.sleep(1)

        if save_video:
            sub_save_path_video = os.path.join(save_path_video, "input")
            os.makedirs(sub_save_path_video, exist_ok=True)
            vis.capture_screen_image(os.path.join(sub_save_path_video, '%d.png' %num_rotate))
        num_rotate += 1
        if num_rotate == 360:
            break

    vis.destroy_window()
    if save_video:
        video_make(sub_save_path_video, idx)

def vis_sample(gt_ho, in_ho, out_ho, idx, mje_in=None, mje_out=None, vis_3d=False, save_video=False):
    hand_gt, obj_gt = gt_ho.get_o3d_meshes(hand_contact=True, obj_contact=True, normalize_pos=True)
    hand_in, obj_in = in_ho.get_o3d_meshes(hand_contact=True, obj_contact=True ,normalize_pos=True)
    hand_in.translate((0.0, 0.2, 0.0))
    obj_in.translate((0.0, 0.2, 0.0))

    if not args.split == 'honn':
        out_ho.hand_contact = in_ho.hand_contact
        out_ho.obj_contact = in_ho.obj_contact

    hand_out, obj_out = out_ho.get_o3d_meshes(hand_contact=True, obj_contact=True, normalize_pos=True)
    hand_out.translate((0.0, 0.4, 0.0))
    obj_out.translate((0.0, 0.4, 0.0))

    # geom_list = [hand_gt, obj_gt, hand_out, obj_out, hand_in, obj_in]
    geom_list = [hand_out, obj_out]
    # hand_contact_gt = util.val_to_class(torch.Tensor(in_ho.hand_contact).cuda()) 
    # obj_contact_gt = util.val_to_class(torch.Tensor(in_ho.obj_contact).cuda()) 
    # obj_gt_freq, hand_gt_freq = util.get_gt_distribution(hand_contact_gt.unsqueeze(0), obj_contact_gt.unsqueeze(0))
    # print(obj_gt_freq, hand_gt_freq)
    geom_list.append(util.text_3d('In', pos=[-0.4, 0.2, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('Refined', pos=[-0.4, 0.4, 0], font_size=40, density=2))
    geom_list.append(util.text_3d('GT', pos=[-0.4, 0.0, 0], font_size=40, density=2))

    if mje_in is not None:
        geom_list.append(util.text_3d('MJE in {:.2f}cm out {:.2f}cm'.format(mje_in * 100, mje_out * 100), pos=[-0.4, -0.2, 0], font_size=40, density=2))

    if not vis_3d:
        o3dv.draw_geometries(geom_list)
    else:
        plot_3d(hand_gt, obj_gt, hand_out, obj_out, hand_in, obj_in, idx, save_video)


def calc_mean_dicts(all_dicts, phase=''):
    keys = all_dicts[0].keys()
    mean_dict = dict()
    stds = ['intersection_volume_ho']

    for k in keys:
        l = list()
        for d in all_dicts:
            l.append(d[k])
        mean_dict[k] = np.array(l).mean()

        if k in stds:
            mean_dict[k + '_std'] = util.calc_std(np.array(l))

    return mean_dict


def calc_sample(ho_test, ho_gt, idx, phase='nophase'):
    stats = geometric_eval.geometric_eval(copy.deepcopy(ho_test), copy.deepcopy(ho_gt))

    return stats


def process_sample(sample, idx):
    gt_ho, in_ho, out_ho = sample['gt_ho'], sample['in_ho'], sample['out_ho']
    in_stats = calc_sample(in_ho, gt_ho, idx, 'before ContactOpt')
    out_stats = calc_sample(out_ho, gt_ho, idx, 'after ContactOpt')

    return in_stats, out_stats


def run_eval(args):
    in_file = 'data/optimized_{}.pkl'.format(args.split)
    if args.compressed_storage:
        runs = []
        with open(in_file, 'rb') as f :
            flag = pickle.load(f)
            data_num = args.subfile_num + 1 if flag['rest_flag'] else args.subfile_num
            for _ in range(data_num):
                runs += pickle.load(f) 
            # print(runs)
            f.close()
    else:
        runs = pickle.load(open(in_file, 'rb'))
    print('Loaded {} len {}'.format(in_file, len(runs)))

    # if args.vis or args.physics:
    #     print('Shuffling!!!')
    #     random.shuffle(runs)

    if args.partial > 0:
        runs = runs[:args.partial]

    do_parallel = not args.vis
    if do_parallel:
        all_data = Parallel(n_jobs=mp.cpu_count() - 2)(delayed(process_sample)(s, idx) for idx, s in enumerate(tqdm(runs)))
        in_all = [item[0] for item in all_data]
        out_all = [item[1] for item in all_data]
    else:
        all_data = []   # Do non-parallel
        # vis_idx = [13,27,30,33,35,52,63,84,85,182,244,294,332,360]
        # vis_idx = [151,222,269,688]
        # vis_idx = [4,64,355]
        vis_idx = [107]
        for idx, s in enumerate(tqdm(runs)):
            # if idx <80:
            #     continue
            if idx in vis_idx:
                all_data.append(process_sample(s, idx))

            # if (all_data[-1][0]['unalign_hand_joints'] - all_data[-1][1]['unalign_hand_joints'] >= 0.000) and all_data[-1][1]['unalign_hand_joints'] <= 0.01:
            # if all_data[-1][1]['unalign_hand_joints'] <= 0.003:
            # if idx >= 80 and idx<=98 :
                if args.vis:
                    print('In vs GT\n', pprint.pformat(all_data[-1][0]))
                    print('Out vs GT\n', pprint.pformat(all_data[-1][1]))
                    if args.split == 'im_pred_trans':
                        vis_sample(s['gt_ho'], s['in_ho'], s['out_ho'], idx, mje_in=all_data[-1][0]['objalign_hand_joints'], mje_out=all_data[-1][1]['objalign_hand_joints'], vis_3d=False, save_video=True)
                    else:
                        print(idx)
                        vis_sample(s['gt_ho'], s['in_ho'], s['out_ho'], idx, mje_in=all_data[-1][0]['unalign_hand_joints'], mje_out=all_data[-1][1]['unalign_hand_joints'], vis_3d=False, save_video=False)
                    # vis_3d: visualization with 3D rotation
            

        in_all = [item[0] for item in all_data]
        out_all = [item[1] for item in all_data]

    mean_in = calc_mean_dicts(in_all, 'In vs GT')
    mean_out = calc_mean_dicts(out_all, 'Out vs GT')
    if "Contact_IoU" in mean_in.keys():
        mean_out['Contact_IoU'] = mean_in['Contact_IoU'] # Contact_IoU measures the difference between the groundtruth contact map and the predicted contact map.
        # However, out_all records the target contactmap as predicted contactmap, the values need to be adjusted.
    print('In vs GT\n', pprint.pformat(mean_in))
    print('Out vs GT\n', pprint.pformat(mean_out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run eval on fitted pkl')
    parser.add_argument('--split', default='fine', type=str)
    parser.add_argument('--vis', action='store_true')
    # parser.add_argument('--vis', default=True)
    parser.add_argument('--contact_f1', action='store_true')
    parser.add_argument('--pen', action='store_true')
    parser.add_argument('--saveobj', action='store_true')
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n samples')
    parser.add_argument('--subfile_num', default=8, type=int, help='Set the number of sub-files to store')
    parser.add_argument('--compressed_storage', default=False, type=bool, help='Segmented storage for smaller running memory')
    args = parser.parse_args()

    start_time = time.time()
    run_eval(args)
    print('Eval time', time.time() - start_time)
