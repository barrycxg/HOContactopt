# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("/home/barry/cxg/HOCopt")
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import hocopt.arguments as arguments
from hocopt.cren_net import CREN
from tqdm import tqdm
import hocopt.util as util
from hocopt.loader import ContactDBDataset
import torch.nn.functional as F


def calc_losses(network_out, contact_obj_gt, contact_hand_gt, sampled_verts_idx, epoch):
    losses = dict()

    batch_size = contact_obj_gt.shape[0]
    batch = torch.zeros(sampled_verts_idx.shape, device=device, dtype=torch.long)
    for i in range(batch_size):
        batch[i, :] = i
    batch = batch.view(-1)
    contact_obj_gt = contact_obj_gt[batch, sampled_verts_idx.view(-1), :]   # Select sampled verts
    contact_obj_gt = contact_obj_gt.reshape(batch_size, sampled_verts_idx.shape[1], 1)  # Reshape into network's shape

    class_hand_gt = util.val_to_class(contact_hand_gt).squeeze(2)
    class_obj_gt = util.val_to_class(contact_obj_gt).squeeze(2)
    # print('class obj gt', class_obj_gt.shape, network_out['contact_obj'], class_obj_gt)

    # compute distribution feature
    obj_gt_freq, hand_gt_freq = util.get_gt_distribution(class_hand_gt.clone().detach(), class_obj_gt.clone().detach())

    # add entropy weight without backward
    w_obj, w_hand = util.get_entropy_weight(class_obj_gt.clone().detach().cpu().numpy(), class_hand_gt.clone().detach().cpu().numpy())
    w_obj = w_obj.expand_as(network_out['contact_obj_weight'])
    w_hand = w_hand.expand_as(network_out['contact_hand_weight'])
    
    # network_out['contact_obj'] = network_out['contact_obj'] * w_obj
    # network_out['contact_hand'] = network_out['contact_hand'] * w_hand
    network_out['contact_obj'] = F.log_softmax(network_out['contact_obj'], dim=-1) # turn into the original network output format
    network_out['contact_hand'] = F.log_softmax(network_out['contact_hand'], dim=-1) # turn into the original network output format

    # calculate decay factor
    # scale <- (1-timestep/timesteps)(initial_scale-final_scale)+final_scale 
    if epoch <= args.decay_epochs:
        scale = (1 - epoch / args.decay_epochs) * (args.initial_scale - args.final_scale) + args.final_scale
    else:
        scale = args.final_scale

    losses['contact_obj'] = criterion(network_out['contact_obj'].permute(0, 2, 1), class_obj_gt)
    losses['contact_hand'] = criterion(network_out['contact_hand'].permute(0, 2, 1), class_hand_gt)
    # input prediction: (minibatch, C, d1,d2,...,dk), di is case, C=number of classes, target: (minibatch, di,d2,...,dk)
    losses['distribution_obj'] = criterion2(network_out['dis_feat_obj'], obj_gt_freq)
    losses['distribution_hand'] = criterion2(network_out['dis_feat_hand'], hand_gt_freq)
    # losses['contact_obj_weight'] = torch.sqrt(criterion2(network_out['contact_obj_weight'], w_obj)) # Root Mean Squared Error(RMSE)
    # losses['contact_hand_weight'] = torch.sqrt(criterion2(network_out['contact_hand_weight'], w_hand)) # Root Mean Squared Error(RMSE)

    # Root Mean Squared Logarithmic Error(RMSLE) (option)
    w_obj = torch.log(w_obj + 1)
    w_hand = torch.log(w_hand + 1)
    network_out['contact_obj_weight'] = torch.log(network_out['contact_obj_weight'] + 1)
    network_out['contact_hand_weight'] = torch.log(network_out['contact_hand_weight'] + 1)
    losses['contact_obj_weight'] = torch.sqrt(criterion2(network_out['contact_obj_weight'], w_obj)) * scale
    losses['contact_hand_weight'] = torch.sqrt(criterion2(network_out['contact_hand_weight'], w_hand)) * scale

    return losses, scale


def train_epoch(epoch):
    model.train()
    scheduler.step()
    loss_meter = util.AverageMeter('Loss', ':.2f')

    for idx, data in enumerate(tqdm(train_loader)):
        data = util.dict_to_device(data, device)
        batch_size = data['hand_pose_gt'].shape[0]

        optimizer.zero_grad()
        out = model(data['hand_verts_aug'], data['hand_feats_aug'], data['obj_sampled_verts_aug'], data['obj_feats_aug'])
        losses, scale = calc_losses(out, data['obj_contact_gt'], data['hand_contact_gt'], data['obj_sampled_idx'], epoch)
        loss = losses['contact_obj'] * args.loss_c_obj + losses['contact_hand'] * args.loss_c_hand + losses['distribution_obj'] * args.loss_distri_obj + losses['distribution_hand'] * args.loss_distri_hand + losses['contact_obj_weight'] * args.loss_weight_obj + losses['contact_hand_weight'] * args.loss_weight_hand
        # loss = losses['contact_obj'] * args.loss_c_obj + losses['contact_hand'] * args.loss_c_hand 

        loss_meter.update(loss.item(), batch_size)   # TODO better loss monitoring
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print('{} / {}'.format(idx, len(train_loader)), loss_meter)

            global_iter = epoch * len(train_loader) + idx
            writer.add_scalar('training/loss_contact_obj', losses['contact_obj'], global_iter)
            writer.add_scalar('training/loss_contact_hand', losses['contact_hand'], global_iter)
            writer.add_scalar('training/loss_distribution_obj', losses['distribution_obj'], global_iter)
            writer.add_scalar('training/loss_distribution_hand', losses['distribution_hand'], global_iter)
            # writer.add_scalar('training/loss_contact_obj_weight', losses['contact_obj_weight'], global_iter)
            # writer.add_scalar('training/loss_contact_hand_weight', losses['contact_hand_weight'], global_iter)
            writer.add_scalar('training/lr', scheduler.get_lr(), global_iter)

    print('Train epoch: {}. Avg loss {:.4f} --------------------'.format(epoch, loss_meter.avg))
    print('Decay scale is %f' %(scale))


def test():
    model.eval()

    for idx, data in enumerate(test_loader):
        data = util.dict_to_device(data, device)

        with torch.no_grad():
            out = model(data['hand_verts_aug'], data['hand_feats_aug'], data['obj_sampled_verts_aug'], data['obj_feats_aug'])
            losses, _ = calc_losses(out, data['obj_contact_gt'], data['hand_contact_gt'], data['obj_sampled_idx'], epoch)

    global_iter = epoch * len(train_loader)
    writer.add_scalar('testing/loss_contact_obj', losses['contact_obj'], global_iter)
    writer.add_scalar('testing/loss_contact_hand', losses['contact_hand'], global_iter)
    # writer.add_scalar('testing/loss_distribution_obj', losses['distribution_obj'], global_iter)
    # writer.add_scalar('testing/loss_distribution_hand', losses['distribution_hand'], global_iter)
    # writer.add_scalar('testing/loss_contact_obj_weight', losses['contact_obj_weight'], global_iter)
    # writer.add_scalar('testing/loss_contact_hand_weight', losses['contact_hand_weight'], global_iter)

    # print('Test epoch: Mean joint err {:.2f} cm --------------------'.format(joint_err_meter.avg))


if __name__ == '__main__':
    util.hack_filedesciptor()
    args = arguments.train_network_parse_args()

    train_dataset = ContactDBDataset(args.train_dataset, train=True)
    test_dataset = ContactDBDataset(args.test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=ContactDBDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=ContactDBDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CREN().to(device)

    if args.checkpoint != '':
        print('Attempting to load checkpoint file:', args.checkpoint)
        pretrained_dict = torch.load(args.checkpoint)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'mano' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.optimizer == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) # using adamW optimizer
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    bin_weights = torch.Tensor(np.loadtxt(util.AFFORDANCE_BIN_WEIGHTS_FILE)).to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=bin_weights)
    criterion = torch.nn.NLLLoss(weight=bin_weights)
    criterion2 = torch.nn.MSELoss(reduction='mean')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8], gamma=0.1)  # TODO automatic?
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    writer = SummaryWriter(logdir='runs/' + args.desc)
    writer.add_text('Hyperparams', args.all_str, 0)

    for epoch in range(1, args.epochs):
        train_epoch(epoch)
        test()
        torch.save(model.state_dict(), 'checkpoints/{}.pt'.format(args.desc))
        print('\n')

