import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
import json
import bisect
import clip
import random


class DMPDatasetEERandTarXYLang(Dataset):
    def __init__(self, data_dirs, random=True, normalize='separate', length_total=91, depth_scale=1000.):
        # |--datadir
        #     |--trial0
        #         |--img0
        #         |--img1
        #         |--imgx
        #         |--states.json
        #     |--trial1
        #     |--...

        assert normalize in ['none']

        all_dirs = []
        for data_dir in data_dirs:
            all_dirs = all_dirs + [ f.path for f in os.scandir(data_dir) if f.is_dir() ]

        self.random = random
        self.normalize = normalize
        self.length_total = length_total
        self.trials = []
        self.lengths_index = []


        length = 0
        for trial in all_dirs:

            # trial_id = int(trial.strip().split(r'/')[-1])
            # if not ((trial_id >= 1700) and (trial_id < 1725)):
            #     continue

            trial_dict = {}

            states_json = os.path.join(trial, 'states.json')
            with open(states_json) as json_file:
                states_dict = json.load(json_file)
                json_file.close()
            
            # There are (trial_dict['len']) states
            trial_dict['len'] = len(states_dict)
            trial_dict['img_paths'] = [os.path.join(trial, str(i) + '.png') for i in range(trial_dict['len'])]
            trial_dict['joint_angles'] = np.asarray([self._joints_to_sin_cos_(states_dict[i]['joints']) for i in range(trial_dict['len'])])
            trial_dict['sentence'] = states_dict[0]['sentence']
            
            # There are (trial_dict['len']) steps in the trial, which means (trial_dict['len'] + 1) states
            trial_dict['len'] -= 1
            self.trials.append(trial_dict)
            length = length + trial_dict['len']
            self.lengths_index.append(length)


    def _joints_to_sin_cos_(self, joints):
        sin_cos_joints = [0] * 8
        for i in range(len(joints)):
            sin_cos_joints[i * 2] = np.sin(joints[i])
            sin_cos_joints[i * 2 + 1] = np.cos(joints[i])
        return sin_cos_joints

    def __len__(self):
        return self.lengths_index[-1]

    def __getitem__(self, index):
        trial_idx = bisect.bisect_right(self.lengths_index, index)
        if trial_idx == 0:
            step_idx = index
        else:
            step_idx = index - self.lengths_index[trial_idx - 1]


        img = torch.tensor(
            io.imread(self.trials[trial_idx]['img_paths'][step_idx])[::-1,:,:3] \
                / 255, dtype=torch.float32)


        sentence = self.trials[trial_idx]['sentence']
        sentence = clip.tokenize([sentence])[0]

        # if self.normalize == 'separate':
        #     joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints) / (self.var_joints ** (1/2)), dtype=torch.float32)
        #     joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints) / (self.var_joints ** (1/2)), dtype=torch.float32)
        # elif self.normalize == 'together':
        #     joint_angles = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx] - self.mean_joints_together) / (self.var_joints_together ** (1/2)), dtype=torch.float32)
        #     joint_angles_traj = torch.tensor((self.trials[trial_idx]['joint_angles'][step_idx:] - self.mean_joints_together) / (self.var_joints_together ** (1/2)), dtype=torch.float32)
        if self.normalize == 'none':
            joint_angles_traj = torch.tensor(self.trials[trial_idx]['joint_angles'][step_idx:], dtype=torch.float32)


        length_total = self.length_total
        length_left = max(length_total - joint_angles_traj.shape[0], 0)

        if length_left > 0:
            # ee_traj_appendix = ee_traj[-1:].repeat(length_left, 1)
            # ee_traj = torch.cat((ee_traj, ee_traj_appendix), axis=0)

            joint_angles_traj_appendix = joint_angles_traj[-1:].repeat(length_left, 1)
            joint_angles_traj = torch.cat((joint_angles_traj, joint_angles_traj_appendix), axis=0)

            # displacement_traj_appendix = displacement_traj[-1:].repeat(length_left, 1)
            # displacement_traj = torch.cat((displacement_traj, displacement_traj_appendix), axis=0)
        else:
            # ee_traj = ee_traj[:length_total]
            joint_angles_traj = joint_angles_traj[:length_total]
            # displacement_traj = displacement_traj[:length_total]

        # print(joint_angles_traj)
        phis = torch.tensor(np.linspace(0.0, 1.0, length_total, dtype=np.float32))
        mask = torch.ones(phis.shape)

        # print(joint_angles_traj)
        return img, phis, mask, sentence, joint_angles_traj
        # return img, joint_angles, ee_pos, ee_traj, ee_xy, length, target_pos, phis, mask, target_xy, sentence[0], joint_angles_traj, displacement#, displacement_traj


def pad_collate_xy_lang(batch):
    (img, phis, mask, sentence, joint_angles_traj) = zip(*batch)

    img = torch.stack(img)
    sentence = torch.stack(sentence)
    # print(sentence.shape)
    joint_angles_traj = torch.nn.utils.rnn.pad_sequence(joint_angles_traj, batch_first=True, padding_value=0)
    joint_angles_traj = torch.transpose(joint_angles_traj, 1, 2)
    phis = torch.nn.utils.rnn.pad_sequence(phis, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, joint_angles_traj.shape[1], 1)
    mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0).unsqueeze(1).repeat(1, joint_angles_traj.shape[1], 1)

    return img, phis, mask, sentence, joint_angles_traj


if __name__ == '__main__':
    data_dirs = [
        '/home/local/ASUAD/yzhou298/Documents/projects/intro_paper/tiny_ur5/collected3'
    ]
    dataset = DMPDatasetEERandTarXYLang(data_dirs, random=False, length_total=50, normalize='none')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3,
                                          shuffle=True, num_workers=1,
                                          collate_fn=pad_collate_xy_lang)

    for img, phis, mask, sentence, joint_angles_traj in dataloader:
        # print(target, joint_angles, ee_pos, ee_traj, length, target_pos)
        # print(length, len(ee_traj))
        print(img.shape, phis.shape, mask.shape, sentence.shape, joint_angles_traj.shape)

        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img[0].numpy())

        ax = fig.add_subplot(1, 2, 2)
        xs = np.arange(joint_angles_traj.shape[2])
        ax.plot(xs, joint_angles_traj[0, 0, :], label='0')
        ax.plot(xs, joint_angles_traj[0, 1, :], label='1')
        ax.plot(xs, joint_angles_traj[0, 2, :], label='2')
        ax.plot(xs, joint_angles_traj[0, 3, :], label='3')
        ax.legend()
        
        plt.show()
