import numpy as np
import gym
import gym_rubiks_cube


class RandomAgent:
    def predict(self, obs):
        return np.random.randint(18), None


if __name__ == "__main__":
    env = gym.make("RubiksCube-v0")

    model = RandomAgent()

    """testing"""
    reward_sum = 0.0
    obs = env.reset()
    for i in range(0, 10):
        done = False
        while not done:
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

    env.close()
