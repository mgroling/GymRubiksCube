import numpy as np
import gym
import gym_rubiks_cube


class RandomAgent:
    def predict(self, obs):
        return np.random.randint(18), None


if __name__ == "__main__":
    env = gym.make("RubiksCube-v0")

    env.scramble_params = 20  # number of random actions to do for scrambling (called in reset), = 0 don't scramble the cube

    env.screen_width = 400  # you can reduce the screen size if FPS are too low
    env.screen_height = 400

    model = RandomAgent()

    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    env.close()
