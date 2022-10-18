import numpy as np
# np.set_printoptions(precision=3, suppress=True)
from models.film_model import Backbone
from utils.load_data_tiny_ur5 import DMPDatasetEERandTarXYLang, pad_collate_xy_lang
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch
from torch.optim.lr_scheduler import StepLR
import os
import matplotlib.pyplot as plt
import time
import random
import clip
import re


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def train(writer, name, epoch_idx, data_loader, model, 
    optimizer, ckpt_path, save_ckpt, stage):
    model.train()
    criterion2 = nn.HuberLoss(reduction='none')

    for idx, (img, phis, mask, sentence, joint_angles_traj) in enumerate(data_loader):
        global_step = epoch_idx * len(data_loader) + idx

        # Prepare data
        img = img.to(device)
        phis = phis.to(device)
        mask = mask.to(device)
        sentence = sentence.to(device)
        joint_angles_traj = joint_angles_traj.to(device)

        # Forward pass
        optimizer.zero_grad()
        joints_trajectory_pred = model(img, sentence, phis)
        joints_trajectory_pred = joints_trajectory_pred * mask
        
        joint_angles_traj = joint_angles_traj * mask
        weight_matrix = torch.tensor(np.array([1 ** i for i in range(joint_angles_traj.shape[-1])]), dtype=torch.float32) + torch.tensor(np.array([0.9 ** i for i in range(joint_angles_traj.shape[-1]-1, -1, -1)]), dtype=torch.float32)
        weight_matrix = weight_matrix.unsqueeze(0).unsqueeze(1).repeat(joint_angles_traj.shape[0], joint_angles_traj.shape[1], 1).cuda()
        loss = (criterion2(joints_trajectory_pred, joint_angles_traj) * weight_matrix).sum() / (mask * weight_matrix).sum()
        writer.add_scalar('train loss traj', loss.item(), global_step=epoch_idx * len(data_loader) + idx)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # Log and print
        writer.add_scalar('train loss', loss.item(), global_step=epoch_idx * len(data_loader) + idx)
        print(f'epoch {epoch_idx}, step {idx}, l_all {loss.item():.2f}')

        # Save checkpoint
        if save_ckpt:
            if global_step % 10000 == 0:
                if not os.path.isdir(os.path.join(ckpt_path, name)):
                    os.mkdir(os.path.join(ckpt_path, name))
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, name, f'{global_step}.pth'))

    return stage


def test(writer, epoch_idx, data_loader, model, train_dataset_size):
    with torch.no_grad():
        model.eval()
        error_trajectory = 0
        error_gripper = 0
        loss5_accu = 0
        idx = 0
        error_target_position = 0
        error_displacement = 0
        error_ee_pos = 0
        error_joints_prediction = 0
        num_datapoints = 0
        num_trajpoints = 0
        num_grippoints = 0
        criterion2 = nn.MSELoss(reduction='none')

        mean = np.array([ 2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0])
        std = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1]) ** (1/2)
        mean_joints = np.array([-2.26736831e-01, 5.13238925e-01, -1.84928474e+00, 7.77270127e-01, 1.34229937e+00, 1.39107280e-03, 2.12295943e-01])
        std_joints = np.array([1.41245676e-01, 3.07248648e-02, 1.34113984e-01, 6.87947763e-02, 1.41992804e-01, 7.84910314e-05, 5.66411791e-02]) ** (1/2)
        mean_traj_gripper = np.array([2.97563984e-02,  4.47217117e-01,  8.45049397e-02, 0, 0, 0, 0, 0, 0, 2.12295943e-01])
        std_traj_gripper = np.array([4.52914246e-02, 5.01675921e-03, 4.19371463e-03, 1, 1, 1, 1, 1, 1, 5.66411791e-02]) ** (1/2)
        mean_displacement = np.array([2.53345831e-01, 1.14758266e-01, -6.98193015e-02, 0, 0, 0, 0, 0, 0])
        std_displacement = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1])
        std_traj_gripper_centered = np.array([7.16058815e-02, 5.89546881e-02, 6.53571811e-02, 1, 1, 1, 1, 1, 1, 0.23799407366571126])
        
        for idx, (img, phis, mask, sentence, joint_angles_traj) in enumerate(data_loader):
            global_step = epoch_idx * len(data_loader) + idx

            # Prepare data
            img = img.to(device)
            phis = phis.to(device)
            mask = mask.to(device)
            sentence = sentence.to(device)
            joint_angles_traj = joint_angles_traj.to(device)

            # Forward pass0
            joint_angles_traj_pred = model(img, sentence, phis)

            joint_angles_traj_pred = joint_angles_traj_pred * mask
            joint_angles_traj = joint_angles_traj * mask
            # Only training on xyz, ignoring rpy
            # loss1 = criterion2(trajectory_pred, ee_traj).sum() / mask.sum()

            joint_angles_traj_pred = joint_angles_traj_pred.detach().cpu().transpose(2, 1)
            joint_angles_traj = joint_angles_traj.detach().cpu().transpose(2, 1)
            
            error_trajectory_this_time = torch.sum(((joint_angles_traj_pred[:, :, :3] - joint_angles_traj[:, :, :3])) ** 2, axis=2) ** 0.5
            error_trajectory_this_time = torch.sum(error_trajectory_this_time)
            error_trajectory += error_trajectory_this_time
            num_trajpoints += torch.sum(mask[:, :3, :]) / mask.shape[1]


            idx += 1
            print(idx, f'err traj {(error_trajectory / num_trajpoints).item():.4f}')
        writer.add_scalar('test error_trajectory', error_trajectory / num_trajpoints, global_step=epoch_idx * train_dataset_size)



def main(writer, name, batch_size=256):
    # data_root_path = r'/data/Documents/yzhou298'
    # data_root_path = r'/share/yzhou298'
    # data_root_path = r'/share/yzhou298/ckpts/tinyur5'
    ckpt_path = r'/share/yzhou298/ckpts/tinyur5'
    save_ckpt = True
    supervised_attn = True
    curriculum_learning = True
    ckpt = None

    # load model
    model = Backbone(img_size=224, num_traces_out=8, embedding_size=256, num_weight_points=12, input_nc=3)
    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt), strict=True)

    model = model.to(device)

    # load data
    data_dirs = [
        '/share/yzhou298/dataset/tinyur5/collected_long_inst',
        '/share/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate'
    ]
    dataset_train_dmp = DMPDatasetEERandTarXYLang(data_dirs, random=False, length_total=60, normalize='none')
    data_loader_train_dmp = torch.utils.data.DataLoader(dataset_train_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)
    
    data_dirs_val = [
        '/share/yzhou298/dataset/tinyur5/collected_long_inst_val',
        '/share/yzhou298/dataset/tinyur5/collected_long_inst_push_rotate_val'
    ]
    dataset_val_dmp = DMPDatasetEERandTarXYLang(data_dirs_val, random=False, length_total=60, normalize='none')
    data_loader_val_dmp = torch.utils.data.DataLoader(dataset_val_dmp, batch_size=batch_size,
                                          shuffle=True, num_workers=8,
                                          collate_fn=pad_collate_xy_lang)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.HuberLoss()
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    print('loaded')

    # train n epoches
    loss_stage = 0
    for i in range(0, 300):
        whether_test = ((i % 10) == 0)
        loss_stage = train(writer, name, i, data_loader_train_dmp, model, optimizer, ckpt_path, save_ckpt, stage=loss_stage)
        if whether_test:
            test(writer, i + 1, data_loader_val_dmp, model, len(data_loader_train_dmp))


if __name__ == '__main__':
    name = 'train-baseline-bcz-film-resnet-huberloss-long-inst'
    writer = SummaryWriter('runs/' + name)
    main(writer, name)
