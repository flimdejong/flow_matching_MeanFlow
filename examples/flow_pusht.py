#!/usr/bin/env python
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys

sys.dont_write_bytecode = True
sys.path.append('../external/models')
sys.path.append('../external')
import os
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pusht
import torch.nn as nn
from tqdm import tqdm
from unet import ConditionalUnet1D
from resnet import get_resnet
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from termcolor import colored

##################################
########## download the pusht data and put in the folder
dataset_path = './pusht/pusht_cchi_v7_replay.zarr'

obs_horizon = 1
pred_horizon = 16
action_dim = 2
action_horizon = 8
num_epochs = 3001
vision_feature_dim = 514


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def get_dataloader(rank=None, world_size=None, distributed=False):
    """Create dataset and dataloader with optional distributed sampler."""
    dataset = pusht.PushTImageDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    stats = dataset.stats

    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, stats, sampler


def create_networks(device):
    """Create and return the neural networks."""
    vision_encoder = get_resnet('resnet18')
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=vision_feature_dim
    )
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    }).to(device)
    return nets


########################################################################
#### Train the model (single GPU)
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader, stats, _ = get_dataloader(distributed=False)
    nets = create_networks(device)

    sigma = 0.0
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    FM = ConditionalFlowMatcher(sigma=sigma)

    avg_loss_train_list = []
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        for data in tqdm(dataloader):
            x_img = data['image'][:, :obs_horizon].to(device)
            x_pos = data['agent_pos'][:, :obs_horizon].to(device)
            x_traj = data['action'].to(device)

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            # encoder vision features
            image_features = nets['vision_encoder'](x_img.flatten(end_dim=1))
            image_features = image_features.reshape(*x_img.shape[:2], -1)
            obs_features = torch.cat([image_features, x_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            vt = nets['noise_pred_net'](xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

        if epoch == 3000:
            ema.copy_to(nets.parameters())
            PATH = './checkpoint_t/flow_ema_%05d.pth' % epoch
            torch.save({'vision_encoder': nets.vision_encoder.state_dict(),
                        'noise_pred_net': nets.noise_pred_net.state_dict(),
                        }, PATH)
            ema.restore(nets.parameters())


########################################################################
#### Train the model (multi-GPU with DDP)
def train_ddp(rank, world_size):
    """Distributed training function for each GPU."""
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    dataloader, stats, sampler = get_dataloader(rank=rank, world_size=world_size, distributed=True)
    nets = create_networks(device)

    # Wrap model with DDP
    nets = DDP(nets, device_ids=[rank], find_unused_parameters=True)

    sigma = 0.0
    ema = EMAModel(parameters=nets.parameters(), power=0.75)
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )
    FM = ConditionalFlowMatcher(sigma=sigma)

    avg_loss_train_list = []
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        total_loss_train = 0.0

        if rank == 0:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        else:
            pbar = dataloader

        for data in pbar:
            x_img = data['image'][:, :obs_horizon].to(device)
            x_pos = data['agent_pos'][:, :obs_horizon].to(device)
            x_traj = data['action'].to(device)

            x_traj = x_traj.float()
            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            image_features = nets.module['vision_encoder'](x_img.flatten(end_dim=1))
            image_features = image_features.reshape(*x_img.shape[:2], -1)
            obs_features = torch.cat([image_features, x_pos], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

            vt = nets.module['noise_pred_net'](xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(nets.parameters())

        # Reduce loss across all GPUs
        dist.all_reduce(total_loss_train, op=dist.ReduceOp.SUM)
        avg_loss_train = total_loss_train / (len(dataloader) * world_size)

        if rank == 0:
            avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
            print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

            if epoch == 3000:
                ema.copy_to(nets.parameters())
                os.makedirs('./checkpoint_t', exist_ok=True)
                PATH = './checkpoint_t/flow_ema_%05d.pth' % epoch
                torch.save({
                    'vision_encoder': nets.module.vision_encoder.state_dict(),
                    'noise_pred_net': nets.module.noise_pred_net.state_dict(),
                }, PATH)
                ema.restore(nets.parameters())

    cleanup()


def train_multi_gpu():
    """Launch multi-GPU training."""
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Only one GPU available. Falling back to single GPU training.")
        train()
    else:
        print(f"Launching training on {world_size} GPUs")
        torch.multiprocessing.spawn(
            train_ddp,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )


########################################################################
###### test the model
def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nets = create_networks(device)

    PATH = './flow_pusht.pth'
    state_dict = torch.load(PATH, map_location='cpu')
    ema_nets = nets
    ema_nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
    ema_nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])

    _, stats, _ = get_dataloader(distributed=False)

    max_steps = 300
    env = pusht.PushTImageEnv()

    test_start_seed = 1000
    n_test = 1

    for epoch in range(n_test):
        seed = test_start_seed + epoch
        env.seed(seed)

        for pp in range(10):
            obs, info = env.reset()
            obs_deque = collections.deque(
                [obs] * obs_horizon, maxlen=obs_horizon)
            imgs = [env.render(mode='rgb_array')]
            rewards = list()
            done = False
            step_idx = 0

            with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
                while not done:
                    B = 1
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    with torch.no_grad():
                        image_features = ema_nets['vision_encoder'](x_img)
                        obs_features = torch.cat([image_features, x_pos], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        timehorion = 1
                        for i in range(timehorion):
                            noise = torch.rand(1, pred_horizon, action_dim).to(device)
                            x0 = noise.expand(x_img.shape[0], -1, -1)
                            timestep = torch.tensor([i / timehorion]).to(device)

                            if i == 0:
                                vt = ema_nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + x0)
                            else:
                                vt = ema_nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + traj)

                    naction = traj.detach().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    x_img = x_img[0, :].permute((1, 2, 0))

                    for j in range(len(action)):
                        obs, reward, done, _, info = env.step(action[j])
                        obs_deque.append(obs)
                        rewards.append(reward)
                        imgs.append(env.render(mode='rgb_array'))

                        step_idx += 1
                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx > max_steps:
                            done = True
                        if done:
                            import imageio
                            print(f'Score: {max(rewards)}')
                            imageio.mimsave('vis_test.mp4', imgs, fps=30)
                            print("Video saved to vis_test.mp4")
                            break


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'train_multi', 'test', or 'unittest'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'train':
        train()
    elif arg == 'train_multi':
        train_multi_gpu()
    elif arg == 'test':
        test()
    elif arg == 'unittest':
        print("Uni Test Successful")
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'train_multi', 'test', or 'unittest'.")
