# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append("/home/barry/cxg/HOCopt/")
import torch
import pytorch3d
import time
from hocopt.loader import *
from manopth.manolayer import ManoLayer
from manopth import rodrigues_layer
import hocopt.diffcontact as calculate_contact
import hocopt.util as util
from hocopt.hand_object import HandObject
from hocopt.visualize import show_optimization
import torch.autograd as autograd


def optimize_pose(data, hand_contact_target, obj_contact_target, n_iter=250, lr=0.01, w_cont_hand=2, w_cont_obj=1,
                  save_history=False, ncomps=15, w_cont_asym=2, w_opt_trans=0.3, w_opt_pose=1, w_opt_rot=1,
                  caps_top=0.0005, caps_bot=-0.001, caps_rad=0.001, caps_on_hand=False,
                  contact_norm_method=0, w_pen_cost=600, w_obj_rot=0, pen_it=0): # TODO verify each param meaning
    """Runs differentiable optimization to align the hand with the target contact map.
    Minimizes the loss between ground truth contact and contact calculated with DiffContact"""
    batch_size = data['hand_pose_aug'].shape[0]
    device = data['hand_pose_aug'].device

    opt_vector = torch.zeros((batch_size, ncomps + 6 + 3), device=device)   # 3 hand rot, 15 dim poses, 3 hand trans, 3 obj rot
    # opt_vector[:, :18] = data['hand_pose_aug'] # way 2
    opt_vector.requires_grad = True

    mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=ncomps, side='right', flat_hand_mean=False).to(device)

    if data['obj_sampled_idx'].numel() > 1:
        obj_normals_sampled = util.batched_index_select(data['obj_normals_aug'], 1, data['obj_sampled_idx']) # surface normal
    else:   # If we're optimizing over all verts
        obj_normals_sampled = data['obj_normals_aug']

    optimizer = torch.optim.AdamW([opt_vector], lr=lr, amsgrad=True)  # AMSgrad helps
    loss_criterion = torch.nn.L1Loss(reduction='none')  # Benchmarked, L1 performs best vs MSE/SmoothL1
    loss_criterion2 = torch.nn.MSELoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[215], gamma=0.2)
    opt_state = []
    is_thin = mesh_is_thin(data['mesh_aug'].num_verts_per_mesh())
    # print('is thin', is_thin, data['mesh_aug'].num_verts_per_mesh())
    # pytorch3d.num_verts_per_mesh: Return a 1D tensor x with length equal to the number of meshes giving the number of vertices in each mesh.

    for it in range(n_iter):
        optimizer.zero_grad()
        assert not torch.isnan(opt_vector).any(), "opt_vector parameters contain NaN!"
        
        #------ add optimized vector to hand pose and get new hand verts and joints in Mano coordinate system ------#
        # opt vector is optimized variation for hand pose estimation. 
        # Hence the pose estimation need to add hand pose before iterative optimization.
        mano_pose_out = torch.cat([opt_vector[:, 0:3] * w_opt_rot, opt_vector[:, 3:ncomps+3] * w_opt_pose], dim=1)
        mano_pose_out[:, :18] += data['hand_pose_aug'] # combine the optimized vector with origin hand poses (Comment out in way 2)
        tform_out = util.translation_to_tform(opt_vector[:, ncomps+3:ncomps+6] * w_opt_trans) # hand trans (batch, 3) --> (batch, 4, 4)

        hand_verts, hand_joints = util.forward_mano(mano_model, mano_pose_out, data['hand_beta_aug'], [data['hand_mTc_aug'], tform_out])   # 2.2ms
        assert not torch.isnan(hand_verts).any(), "hand verts contains NaN!"
        # hand_verts, hand_joints relate mano coordinate system

        #-------- obtain hand normals --------------#
        if contact_norm_method != 0 and not caps_on_hand: # 5 contact norm methods refer to sdf_to_contact function in diffcontact.py
            with torch.no_grad():   # We need to calculate hand normals if using more complicated methods
                mano_mesh = Meshes(verts=hand_verts, faces=mano_model.th_faces.repeat(batch_size, 1, 1))
                hand_normals = mano_mesh.verts_normals_padded()
        else:
            hand_normals = torch.zeros(hand_verts.shape, device=device)

        #------- obtain object normals ------------#
        obj_verts = data['obj_sampled_verts_aug']
        obj_normals = obj_normals_sampled

        # axisang --> rot matrix
        obj_rot_mat = rodrigues_layer.batch_rodrigues(opt_vector[:, ncomps+6:]) 
        obj_rot_mat = obj_rot_mat.view(batch_size, 3, 3)

        if w_obj_rot > 0: # add rot matrix to object verts and normals 
            # In the optimizating process, objects usually remain stationary while optimizing human hand poses(i.e. w_obj_rot=0)
            # TODO In the optimizating process, the optimized obj_rot_mat is not added the mesh of the object. The final obj_rot_mat is used for optimized obj_rot_mat.
            obj_verts = util.apply_rot(obj_rot_mat, obj_verts, around_centroid=True)
            obj_normals = util.apply_rot(obj_rot_mat, obj_normals)

        """
        origin optimized method
        """
        # #------ calculate contact capsule distance -------#
        # contact_obj, contact_hand = calculate_contact.calculate_contact_capsule(hand_verts, hand_normals, obj_verts, obj_normals,
        #                       caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method)

        # contact_obj_sub = obj_contact_target - contact_obj # Is obj_contact_target accurate？ 
        # contact_obj_weighted = contact_obj_sub + torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        # loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))

        # contact_hand_sub = hand_contact_target - contact_hand
        # contact_hand_weighted = contact_hand_sub + torch.nn.functional.relu(contact_hand_sub) * w_cont_asym  # Loss for 'missing' contact higher
        # # penalize missing contacts(target contact > value from DiffContact) more heavily than "unexpected" contacts. w_cont_asym is additional penalty factor
        # loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        # loss = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand # TODO verify w_cont_obj, w_cont_hand, w_cont_asym
        
        """
        new optimized method
        """
        #------ calculate contact capsule distance -------#
        contact_obj, contact_hand = calculate_contact.calculate_contact_capsule(hand_verts, hand_normals, obj_verts, obj_normals,
                              caps_top=caps_top, caps_bot=caps_bot, caps_rad=caps_rad, caps_on_hand=caps_on_hand, contact_norm_method=contact_norm_method)

        contact_obj_sub = obj_contact_target - contact_obj # Is obj_contact_target accurate？ 
        # ----- obj contact loss weight -----------------#
        obj_contact_missing = torch.zeros_like(contact_obj_sub)
        obj_contact_unexpected = torch.zeros_like(contact_obj_sub)
        obj_contact_missing[contact_obj_sub > 0] = 1
        obj_contact_unexpected[contact_obj_sub < 0] = 1
        obj_dist_missing_weight = contact_obj_sub * obj_contact_missing
        obj_dist_unexpected_weight = -contact_obj_sub * obj_contact_unexpected
        # ----- obj contact loss -----------------#
        contact_obj_weighted = torch.exp(obj_dist_unexpected_weight) * torch.nn.functional.relu(-contact_obj_sub) + torch.exp(obj_dist_missing_weight)*torch.nn.functional.relu(contact_obj_sub) * w_cont_asym  # Loss for 'missing' contact higher
        loss_contact_obj = loss_criterion(contact_obj_weighted, torch.zeros_like(contact_obj_weighted)).mean(dim=(1, 2))
        
        contact_hand_sub = hand_contact_target - contact_hand
        # ----- hand contact loss weight ----------#
        hand_contact_missing = torch.zeros_like(contact_hand_sub)
        hand_contact_unexpected = torch.zeros_like(contact_hand_sub)
        hand_contact_missing[contact_hand_sub > 0] = 1
        hand_contact_unexpected[contact_hand_sub < 0] = 1

        # contact_hand_pd = util.load_contact_zones(sort=False) 
        # merged_id = np.concatenate([contact_hand_pd[key] for contact_id, key in enumerate(contact_hand_pd.keys()) if contact_id !=0]) # get hand tips
        # contact_merged_id = np.sort(merged_id)
        # hand_contact_missing[:, contact_merged_id, :] *= 1.5
        # hand_contact_unexpected[:, contact_merged_id, :] *= 1.0

        hand_dist_missing_weight = contact_hand_sub * hand_contact_missing
        hand_dist_unexpected_weight = -contact_hand_sub * hand_contact_unexpected
        # ----- hand contact loss -----------------#
        contact_hand_weighted = torch.exp(hand_dist_unexpected_weight) * torch.nn.functional.relu(-contact_hand_sub) + torch.exp(hand_dist_missing_weight) * torch.nn.functional.relu(contact_hand_sub) * w_cont_asym # Loss for 'missing' contact higher
        # penalize missing contacts(target contact > value from DiffContact) more heavily than "unexpected" contacts. w_cont_asym is additional penalty factor
        loss_contact_hand = loss_criterion(contact_hand_weighted, torch.zeros_like(contact_hand_weighted)).mean(dim=(1, 2))

        # if it ==240:
        #     print(loss_contact_obj)
        #     print("*********")
        #     print(loss_contact_hand)
        #     print("^^^^^^^^^^^^^^^^^^^^^^^^")

        loss = loss_contact_obj * w_cont_obj + loss_contact_hand * w_cont_hand # TODO verify w_cont_obj, w_cont_hand, w_cont_asym
        # loss = loss_contact_hand * w_cont_hand # TODO verify w_cont_obj, w_cont_hand, w_cont_asym
 
        if w_pen_cost > 0 and it >= pen_it:
            # contact_hand_pd_idx = util.load_contact_zones()
            contact_hand_pd = util.load_contact_zones(sort=False) 
            contact_merged_id = np.concatenate([contact_hand_pd[key] for contact_id, key in enumerate(contact_hand_pd.keys()) if contact_id !=0]) # get hand tips
            contact_hand_pd_idx = np.sort(contact_merged_id)

            # pen_cost, attract_cost = calculate_contact.calculate_verts_cost(hand_verts, hand_normals, data['obj_sampled_verts_aug'], obj_normals_sampled, is_thin, contact_norm_method, contact_hand_pd_idx)
            pen_cost, attract_cost = calculate_contact.calculate_verts_cost(hand_verts, hand_normals, obj_verts, obj_normals, is_thin, contact_norm_method, contact_hand_pd_idx)
            loss += pen_cost.mean(dim=1) * w_pen_cost + attract_cost.mean(dim=1) * 10
            # loss += attract_cost.mean(dim=1) * 10
            
        # if it ==240:
        #     print(attract_cost.mean(dim=1) * 10)
        #     print("---------")
        #     print(pen_cost.mean(dim=1)* w_pen_cost)

        out_dict = {'loss': loss.detach().cpu()}
        if save_history:
            out_dict['hand_verts'] = hand_verts.detach().cpu()#.numpy()
            out_dict['hand_joints'] = hand_joints.detach().cpu()#.numpy()
            out_dict['contact_obj'] = contact_obj.detach().cpu()#.numpy()
            out_dict['contact_hand'] = contact_hand.detach().cpu()#.numpy()
            out_dict['obj_rot'] = obj_rot_mat.detach().cpu()#.numpy()
        opt_state.append(out_dict)

        assert not torch.isnan(loss).any(), "Loss contains NaN!"

        with autograd.detect_anomaly(): # check whetehr loss.backward() contain NAN 
             loss.mean().backward()
        optimizer.step()
        scheduler.step()

    tform_full_out = util.aggregate_tforms([data['hand_mTc_aug'], tform_out])
    return mano_pose_out, tform_full_out, obj_rot_mat, opt_state


def show_optimization_video(data, device):
    """Displays video of optimization process of hand converging"""
    data_gpu = util.dict_to_device(data, device)
    contact_obj_pred = util.batched_index_select(data_gpu['obj_contact_gt'], 1, data_gpu['obj_sampled_idx'])

    out_pose, out_tform, obj_rot_mat, opt_state = optimize_pose(data_gpu, data_gpu['hand_contact_gt'], contact_obj_pred, save_history=True)

    show_optimization(data, opt_state, hand_contact_target=data['hand_contact_gt'], obj_contact_target=contact_obj_pred.detach().cpu(), is_video=True, vis_method=3)


if __name__ == '__main__':
    """Show a video optimization from perturbed pose"""
    test_dataset = ContactDBDataset('data/perturbed_contactpose_handoff_test.pkl')
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=ContactDBDataset.collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx, data in enumerate(dataloader):
        show_optimization_video(data, device)   # do optimization and show video

        if idx >= 10:
            break
