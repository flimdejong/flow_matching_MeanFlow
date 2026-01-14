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
# sys.path.append('../pusht')
import numpy as np
import torch
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

##################################
########## download the pusht data and put in the folder
dataset_path = './pusht/pusht_cchi_v7_replay.zarr'

obs_horizon = 1
pred_horizon = 16
action_dim = 2
action_horizon = 8
num_epochs = 3001
vision_feature_dim = 514

# create dataset from file
dataset = pusht.PushTImageDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

##################################################################
# create network object
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

##################################################################
sigma = 0.0
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)


########################################################################
#### Train the model
def train():
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
###### test the model
def test():
    PATH = './flow_pusht.pth'
    state_dict = torch.load(PATH, map_location='cpu')
    ema_nets = nets
    ema_nets.vision_encoder.load_state_dict(state_dict['vision_encoder'])
    ema_nets.noise_pred_net.load_state_dict(state_dict['noise_pred_net'])

    max_steps = 500
    env = pusht.PushTImageEnv()

    test_start_seed = 1000
    n_test = 1

    ###### please choose the seed you want to test
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

            # List to store all action distribution iterations for the episode
            action_dist_episode = []

            with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
                while not done:
                    B = 1
                    x_img = np.stack([x['image'] for x in obs_deque])
                    x_pos = np.stack([x['agent_pos'] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats['agent_pos'])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    # infer action
                    with torch.no_grad():
                        # get image features
                        image_features = ema_nets['vision_encoder'](x_img)
                        obs_features = torch.cat([image_features, x_pos], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        timehorion = 3
                        # Reset action_dist_iter for each flow-matching prediction
                        action_dist_iter = []
                        
                        for i in range(timehorion):
                            noise = torch.rand(1, pred_horizon, action_dim).to(device)
                            x0 = noise.expand(x_img.shape[0], -1, -1)
                            timestep = torch.tensor([i / timehorion]).to(device)

                            if i == 0:
                                # Store initial noise (unnormalized for visualization)
                                noise_unnorm = pusht.unnormalize_data(
                                    x0[0].detach().cpu().numpy(), stats=stats['action'])
                                action_dist_iter.append(noise_unnorm.copy())
                                
                                vt = nets['noise_pred_net'](x0, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + x0)
                            else:
                                vt = nets['noise_pred_net'](traj, timestep, global_cond=obs_cond)
                                traj = (vt * 1 / timehorion + traj)
                            
                            # Store trajectory after each flow step (unnormalized)
                            traj_unnorm = pusht.unnormalize_data(
                                traj[0].detach().cpu().numpy(), stats=stats['action'])
                            action_dist_iter.append(traj_unnorm.copy())
                        
                        # Store this prediction's flow iterations
                        action_dist_episode.append(action_dist_iter)

                    naction = traj.detach().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats['action'])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    x_img = x_img[0, :].permute((1, 2, 0))

                    # execute action_horizon number of steps
                    for j in range(len(action)):
                        # stepping env
                        obs, reward, done, _, info = env.step(action[j])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
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
                            
                            # Process and visualize action distributions
                            visualize_action_flow(action_dist_episode, action_horizon, obs_horizon)
                            break


def visualize_action_flow(action_dist_episode, action_horizon, obs_horizon):
    """
    Visualize how actions evolve from noise to final predictions.
    
    Args:
        action_dist_episode: List of action_dist_iter for each prediction
        action_horizon: Number of actions actually executed per prediction
        obs_horizon: Observation horizon
    """
    # Step 3: Prune action_dist_episode to only keep executed actions
    # Each prediction executes action_horizon actions starting from index (obs_horizon - 1)
    start_idx = obs_horizon - 1
    end_idx = start_idx + action_horizon
    
    pruned_episode = []
    for action_dist_iter in action_dist_episode:
        # action_dist_iter has shape: [num_flow_steps+1, pred_horizon, action_dim]
        # where num_flow_steps+1 = 4 (noise + 3 flow steps)
        pruned_iter = []
        for step_data in action_dist_iter:
            # Only keep the executed actions
            pruned_iter.append(step_data[start_idx:end_idx, :])
        pruned_episode.append(pruned_iter)
    
    # Step 4: Rearrange so each index belongs to one executed action
    # Current: pruned_episode[prediction_idx][flow_step][action_in_horizon, dim]
    # Target: executed_actions[action_idx][flow_step][dim]
    
    executed_actions = []
    for pred_idx, pred_data in enumerate(pruned_episode):
        num_flow_steps = len(pred_data)  # Should be 4 (noise + 3 steps)
        num_actions_in_pred = pred_data[0].shape[0]  # action_horizon
        
        for action_in_pred in range(num_actions_in_pred):
            action_flow_history = []
            for flow_step in range(num_flow_steps):
                action_flow_history.append(pred_data[flow_step][action_in_pred, :])
            executed_actions.append(action_flow_history)
    
    # Step 5: Create GIF visualization
    create_flow_animation(executed_actions)


def create_flow_animation(executed_actions):
    """
    Create a GIF showing how actions evolve from noise to final values.
    
    Args:
        executed_actions: List where each element is [flow_step][2D action coords]
    """
    num_actions = len(executed_actions)
    num_flow_steps = len(executed_actions[0]) if num_actions > 0 else 0
    
    if num_actions == 0:
        print("No actions to visualize")
        return
    
    # Create color spectrum from yellow to purple
    cmap = plt.cm.plasma  # Yellow to purple colormap
    colors = [cmap(i / max(num_actions - 1, 1)) for i in range(num_actions)]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Get data bounds for consistent axis limits
    all_x = []
    all_y = []
    for action_history in executed_actions:
        for step_data in action_history:
            all_x.append(step_data[0])
            all_y.append(step_data[1])
    
    x_margin = (max(all_x) - min(all_x)) * 0.1 + 1
    y_margin = (max(all_y) - min(all_y)) * 0.1 + 1
    x_lim = (min(all_x) - x_margin, max(all_x) + x_margin)
    y_lim = (min(all_y) - y_margin, max(all_y) + y_margin)
    
    flow_step_names = ['Noise', 'Flow Step 1', 'Flow Step 2', 'Flow Step 3']
    
    def init():
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('Action Dimension 1', fontsize=12)
        ax.set_ylabel('Action Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        return []
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel('Action Dimension 1', fontsize=12)
        ax.set_ylabel('Action Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        flow_step = frame % num_flow_steps
        ax.set_title(f'Action Flow Evolution - {flow_step_names[flow_step]}', fontsize=14)
        
        # Plot all actions at current flow step
        for action_idx, action_history in enumerate(executed_actions):
            x = action_history[flow_step][0]
            y = action_history[flow_step][1]
            ax.scatter(x, y, c=[colors[action_idx]], s=50, alpha=0.7, 
                      edgecolors='black', linewidth=0.5)
        
        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, num_actions - 1))
        sm.set_array([])
        
        # Draw trajectory lines connecting consecutive actions at this flow step
        if num_actions > 1:
            xs = [action_history[flow_step][0] for action_history in executed_actions]
            ys = [action_history[flow_step][1] for action_history in executed_actions]
            ax.plot(xs, ys, 'k-', alpha=0.2, linewidth=0.5)
        
        return []
    
    # Create animation - show each flow step for longer
    frames_per_step = 30  # 30 frames per flow step
    total_frames = num_flow_steps * frames_per_step
    
    def animate_extended(frame):
        return animate(frame // frames_per_step)
    
    anim = FuncAnimation(fig, animate_extended, init_func=init, 
                        frames=total_frames, interval=50, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save('action_flow_evolution.gif', writer=writer)
    print("GIF saved to action_flow_evolution.gif")
    
    plt.close(fig)
    
    # Also create a static comparison plot
    create_static_comparison(executed_actions, colors, flow_step_names)


def create_static_comparison(executed_actions, colors, flow_step_names):
    """Create a static plot showing all flow steps side by side."""
    num_flow_steps = len(executed_actions[0])
    
    fig, axes = plt.subplots(1, num_flow_steps, figsize=(5 * num_flow_steps, 5))
    
    # Get consistent axis limits
    all_x = []
    all_y = []
    for action_history in executed_actions:
        for step_data in action_history:
            all_x.append(step_data[0])
            all_y.append(step_data[1])
    
    x_margin = (max(all_x) - min(all_x)) * 0.1 + 1
    y_margin = (max(all_y) - min(all_y)) * 0.1 + 1
    x_lim = (min(all_x) - x_margin, max(all_x) + x_margin)
    y_lim = (min(all_y) - y_margin, max(all_y) + y_margin)
    
    for step_idx, ax in enumerate(axes):
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_title(flow_step_names[step_idx], fontsize=12)
        ax.set_xlabel('Action Dim 1')
        ax.set_ylabel('Action Dim 2')
        ax.grid(True, alpha=0.3)
        
        for action_idx, action_history in enumerate(executed_actions):
            x = action_history[step_idx][0]
            y = action_history[step_idx][1]
            ax.scatter(x, y, c=[colors[action_idx]], s=30, alpha=0.7,
                      edgecolors='black', linewidth=0.3)
        
        # Draw trajectory
        xs = [ah[step_idx][0] for ah in executed_actions]
        ys = [ah[step_idx][1] for ah in executed_actions]
        ax.plot(xs, ys, 'k-', alpha=0.2, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('action_flow_comparison.png', dpi=150)
    print("Static comparison saved to action_flow_comparison.png")
    plt.close(fig)


if __name__ == '__main__':
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'test', or 'print'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == 'train':
        train()
    elif arg == 'test':
        test()
    elif arg == 'unittest':
        print("Uni Test Successful")
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'test', or 'print'.")
