import numpy as np
import gym
import gym_rubiks_cube
import torch
from torch import nn
from torch import optim


class ReplayBuffer:
    def __init__(self, max_size, env) -> None:
        self.obs_t = np.empty((max_size, *env.observation_space.shape))
        self.obs_tp1 = np.empty((max_size, *env.observation_space.shape))
        self.action = np.empty((max_size,))
        self.reward = np.empty((max_size,))
        self.max_size = max_size
        self.size = 0

    def add(self, obs_t, obs_tp1, action, reward):
        if self.size < self.max_size:
            self.obs_t[self.size] = obs_t
            self.obs_tp1[self.size] = obs_tp1
            self.action[self.size] = action
            self.reward[self.size] = reward
            self.size += 1
        else:
            # randomly replace a sample
            rand = np.random.randint(self.size)
            self.obs_t[rand] = obs_t
            self.obs_tp1[rand] = obs_tp1
            self.action[rand] = action
            self.reward[rand] = reward

    def sample(self, num_samples):
        """randomly sample num_samples from the replay buffer"""
        assert num_samples <= self.size, "Not enough samples in the replay buffer"
        rand = np.random.choice(self.size, num_samples, replace=False)

        return (
            self.obs_t[rand],
            self.obs_tp1[rand],
            self.action[rand],
            self.reward[rand],
        )


class RLAgent(torch.nn.Module):
    env = None

    def __init__(self, env, layers):
        assert (
            len(env.observation_space.shape) == 1
        ), "only environemnts with 1D observation space are accepted"

        super(RLAgent, self).__init__()
        self.flatten = nn.Flatten()
        self.env = env

        temp_layers = []
        for i in range(len(layers) - 1):
            temp_layers.append(nn.Linear(layers[i], layers[i + 1]))
            temp_layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], layers[0]),
            nn.ReLU(),
            *temp_layers,
            nn.Linear(layers[-1], env.action_space.n),
        )

    def forward(self, observation):
        return self.linear_relu_stack(observation)

    def predict(self, observation):
        logits = self.linear_relu_stack(observation)
        pred_prob = nn.Softmax(dim=0)(logits)
        return pred_prob.argmax(0).item()


if __name__ == "__main__":
    BATCH_SIZE = 32
    DISCOUNT = 0.99

    env = gym.make("RubiksCube-v0")
    env.cap_fps = 10
    env.max_steps = 50
    env.scramble_params = 3

    buf = ReplayBuffer(10000, env)
    model = RLAgent(env, [512])
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(1):
        """collect samples"""
        obs = env.reset()
        old_obs = None

        done = False
        while not done:
            action = model.predict(torch.tensor(obs).type(torch.FloatTensor))
            old_obs = obs
            obs, reward, done, _ = env.step(action)
            buf.add(old_obs, obs, action, reward)

        """update model parameters"""
        obs_t, obs_tp1, action, reward = buf.sample(BATCH_SIZE)
        reward_next_step = reward + DISCOUNT * (
            model.forward(torch.tensor(obs_tp1).type(torch.FloatTensor)).max().item()
        )
        pred = (
            model.forward(torch.tensor(obs_t).type(torch.FloatTensor))
            .cpu()
            .detach()
            .numpy()
        )
        pred_copy = pred.copy()
        pred_copy[
            np.arange(len(action)), np.array(action, dtype=np.int32)
        ] = reward_next_step

        l = loss(
            torch.tensor(pred).type(torch.FloatTensor),
            torch.tensor(pred_copy).type(torch.FloatTensor),
        )
        l.backward()

        optimizer.step()

        optimizer.zero_grad()
