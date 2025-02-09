# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime


def parse_dataset(args):
    """ Converts the --split argument into a dataset file """
    if args.split == 'aug' or args.split == 'aug1':
        args.train_dataset = 'data/perturbed_contactpose_train.pkl'
        args.test_dataset = 'data/perturbed_contactpose_test.pkl'
    elif args.split == 'fine':
        args.train_dataset = None
        # args.test_dataset = 'data/contactpose_test.pkl'
        args.test_dataset = 'data/contactpose_handoff_test.pkl'
    elif args.split == 'im_ho3d':
        args.train_dataset = None
        args.test_dataset = 'data/ho3d_image.pkl'
    elif args.split == 'im_dexycb':
        args.train_dataset = None
        args.test_dataset = 'data/dexycb_image.pkl'
    elif args.split == 'demo':
        args.train_dataset = None
        args.test_dataset = 'data/ho3d_image_demo.pkl'
    elif args.split == 'handoff':
        args.train_dataset = 'data/perturbed_contactpose_handoff_train.pkl'
        args.test_dataset = 'data/perturbed_contactpose_handoff_test.pkl'
    else:
        raise ValueError('Unknown dataset')
    
    # print using dataset
    if args.train_dataset is not None:
        print("The training dataset is: %s" %(args.train_dataset))
    if args.test_dataset is not None:
        print("The test dataset is: %s" %(args.test_dataset))


def run_contactopt_parse_args():
    parser = argparse.ArgumentParser(description='Alignment networks training')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--split', default='handoff', type=str) # aug, handoff
    parser.add_argument('--lr', type=float)
    parser.add_argument('--n_iter', type=int)
    parser.add_argument('--partial', default=-1, type=int, help='Only run for n batches')
    parser.add_argument('--w_cont_hand', type=float, help='Weight of the hand contact in optimization')
    parser.add_argument('--sharpen_thresh', type=float)
    parser.add_argument('--ncomps', type=int)
    parser.add_argument('--w_cont_asym', type=float)
    parser.add_argument('--w_opt_trans', type=float)
    parser.add_argument('--w_opt_rot', type=float)
    parser.add_argument('--w_opt_pose', type=float)
    parser.add_argument('--caps_rad', type=float)
    parser.add_argument('--caps_hand', action='store_true')
    parser.add_argument('--cont_method', type=int)
    parser.add_argument('--caps_top', type=float)
    parser.add_argument('--caps_bot', type=float)
    parser.add_argument('--w_pen_cost', type=float)
    parser.add_argument('--pen_it', type=float)
    parser.add_argument('--w_obj_rot', type=float)
    parser.add_argument('--rand_re', type=int)
    parser.add_argument('--rand_re_trans', type=float)
    parser.add_argument('--rand_re_rot', type=float)
    parser.add_argument('--vis_method', type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--min_cont', default=30, type=int, help='Cut grasps with less than this points of initial contact')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--subfile_num', default=8, type=int, help='Set the number of sub-files to store')
    parser.add_argument('--compressed_storage', default=False, type=bool, help='Segmented storage for smaller running memory')
    args = parser.parse_args()
    parse_dataset(args)

    if args.vis:
        args.batch_size = 1

    return args


def train_network_parse_args():
    parser = argparse.ArgumentParser(description='Alignment networks training')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch_size', default=64, type=int) # 32, 64
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--split', default='handoff', type=str) # handoff/aug
    # parser.add_argument('--loss_pose', default=0, type=float)
    parser.add_argument('--loss_c_obj', default=1, type=float)
    parser.add_argument('--loss_c_hand', default=1, type=float)
    parser.add_argument('--loss_distri_obj', default=1, type=float) # 10
    parser.add_argument('--loss_distri_hand', default=1, type=float) # 10
    parser.add_argument('--loss_weight_obj', default=1, type=float)
    parser.add_argument('--loss_weight_hand', default=1, type=float)
    # parser.add_argument('--loss_3d', default=0, type=float)
    parser.add_argument('--epochs', default=121, type=int)
    parser.add_argument('--decay_epochs', default=120, type=int)
    parser.add_argument('--initial_scale', default= 10, type=int)
    parser.add_argument('--final_scale', default= 1, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--desc', default='', type=str)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()

    if args.desc == '':
        args.desc = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    all_str = ''
    for key, val in vars(args).items():
        all_str += '--{}={} '.format(key, val)

    print(all_str)   # Convert to dict and print
    args.all_str = all_str

    parse_dataset(args)

    return args



