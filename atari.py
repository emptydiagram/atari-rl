import math
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# for efficiency, the input is a state s, the output is an array of Q-values,
# Q(s, a), for each of the possible actions a
class DeepQNetworkCNN(nn.Module):
    def __init__(self, in_channels, conv1_hidden_channels, conv2_hidden_channels, fc_hidden_units, num_outputs):
        super().__init__()
        # from 2013 paper:
        #  - 8x8 kernel, 16 feature maps, stride 4, followed by ReLU
        #  - 4x4 kernel, 32 feature maps, stride 2, followed by ReLU
        #  - fully connected hidden layer w/ 256 ReLU neurons

        # (H - K + 1)/S
        H1 = 84
        K1 = 8
        S1 = 4
        H2 = math.ceil((H1 - K1 + 1)/S1)
        K2 = 4
        S2 = 2
        H3 = math.ceil((H2 - K2 + 1)/S2)
        # (H1 = 84) implies (H3 = 9)
        # H1 = 84
        # K1 = 8
        # S1 = 4
        # 0 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64 68 72 76
        # H2 = 20
        # K2 = 4
        # S2 = 2
        # ceil[(20 - 4 + 1)/2] = ceil[17/2] = 9
        # 0 2 4 6 8 10 12 14 16

        self.conv1 = nn.Conv2d(in_channels, conv1_hidden_channels, K1, stride=S1, padding=0)
        self.conv2 = nn.Conv2d(conv1_hidden_channels, conv2_hidden_channels, K2, stride=S2, padding=0)
        self.fc = nn.Linear(conv2_hidden_channels * H3 ** 2, fc_hidden_units)
        self.out = nn.Linear(fc_hidden_units, num_outputs)


    def forward(self, x):
        """Given preprocessed input `x`, where `x` is (by default) a 4x84x84 tensor obtained from a
        stack of four 210x160 pixel Atari game frames `x = (s_{t-3}, s_{t-2}, s_{t-1}, s_t)`, computes for
        every action `a` in A the Q-score `Q(x, a)`, aka the current estimate of taking action `a` in state `x`.

        The value `a* := argmax_{a in A} [ Q(x,a) ]` is the greedy action. The agent using the DQN will (often) use an
        epsilon-greedy strategy (e.g. take a random action with probability `eps`, and `a*` with probability
        `1 - eps`)"""
        print(x.shape)
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = z.reshape(-1)
        z = F.relu(self.fc(z))
        z = self.out(z)
        print(z)
        return z

# Q-learning agent:
# initialize Q: S x A -> R
# for each step of the episode:
#   choose eps-greedy action from Q
class DQNAgent:
    def __init__(self, num_actions):
        self.dqn = DeepQNetworkCNN(in_channels=4, conv1_hidden_channels=16, conv2_hidden_channels=32,
                                   fc_hidden_units=256, num_outputs=num_actions)

    def take_action(self, obs):
        scores = self.dqn(obs)
        act = torch.argmax(scores).item()
        return act



def run_breakout():
    RAND_SEED = 278933

    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env.action_space.seed(RAND_SEED)


    # preprocess. these are all just the defaults
    # also add the customary stack of 4 frames
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    obs, info = env.reset(seed=RAND_SEED)
    print(info)

    dqn_agent = DQNAgent(env.action_space.n)

    for _ in range(1000):
        # Random action
        # action = env.action_space.sample()

        obs_tensor = torch.from_numpy(np.array(obs)).float()
        action = dqn_agent.take_action(obs_tensor)
        obs, rew, terminated, truncated, info = env.step(action)
        # obs is an ndarray of shape
        # print(type(obs))
        # dict
        # print(type(info))

        if terminated or truncated:
            print(f"Resetting because {'terminated' if terminated else 'truncated'}!")
            obs, info = env.reset()

    env.close()

if __name__ == '__main__':
    # TODO kick it off
    print("hello, breakout")
    run_breakout()
