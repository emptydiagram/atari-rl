from collections import deque
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
        z = F.relu(self.conv1(x))
        z = F.relu(self.conv2(z))
        z = z.reshape(-1)
        z = F.relu(self.fc(z))
        z = self.out(z)
        # print(z)
        return z


class ReplayMemory:
    def __init__(self, capacity):
        self.maxlen = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        # print(transition)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q-learning agent:
# initialize Q: S x A -> R
# for each step of the episode:
#   choose eps-greedy action from Q
class DQNAgent:
    def __init__(self, num_actions, replay_mem_size):
        self.dqn = DeepQNetworkCNN(in_channels=4, conv1_hidden_channels=16, conv2_hidden_channels=32,
                                   fc_hidden_units=256, num_outputs=num_actions)

        self.replay_memory = ReplayMemory(replay_mem_size)


    def take_greedy_action(self, obs):
        scores = self.dqn(obs)
        act = torch.argmax(scores).item()
        print(f"{act=}")
        return act

    def add_transition(self, transition):
        self.replay_memory.push(transition)



def train_breakout():
    RAND_SEED = 278933

    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env.action_space.seed(RAND_SEED)

    # from DQN papers
    minibatch_size = 32

    max_num_steps = 10000
    replay_mem_size = 5000

    # epsilon schedule: linear annealing
    eps_anneal_start = 1.0
    eps_anneal_end = 0.1
    # this is 1 million (ish? frames =? steps) in the DQN paper
    eps_anneal_length = 0.5 * max_num_steps

    # eps_start = 0.7
    # eps_end = 0.2
    # eps_end_step = 9
    # t = 0: 0.7
    # t = 4: 0.45
    # t = 9: 0.2

    # 0.7 - (0.5/10) * (t + 1)
    # = 0.1 * (7 - (t + 1)/2)
    #
    # eps_start + (eps_end - eps_start) * (t + 1) / (eps_end_step + 1)

    # preprocess. these are all just the defaults
    # also add the customary stack of 4 frames
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=1, screen_size=84, terminal_on_life_loss=False,
                                          grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    obs, info = env.reset(seed=RAND_SEED)
    print(info)

    dqn_agent = DQNAgent(num_actions=env.action_space.n, replay_mem_size=replay_mem_size)

    for t in range(max_num_steps):
        epsilon =  eps_anneal_start + (eps_anneal_end - eps_anneal_start) * t  / eps_anneal_length
        if t % 10 == 0:
            print(f"\n------ on step {t=}, {epsilon=}")
            print("now, replay memory size = ", len(dqn_agent.replay_memory))

        obs_tensor = torch.from_numpy(np.array(obs)).float()

        # take action
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = dqn_agent.take_greedy_action(obs_tensor)
        new_obs, rew, terminated, truncated, info = env.step(action)

        trans = (obs, action, new_obs, rew)
        dqn_agent.add_transition(trans)
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
    train_breakout()
