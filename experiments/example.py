import numpy as np
import gym
import gym_rubiks_cube


class RandomAgent:
    def predict(self, obs):
        return np.random.randint(18), None


if __name__ == "__main__":
    env = gym.make("RubiksCube-v0")

    env.scramble_params = 20  # number of random actions to do for scrambling (called in reset), =0 don't scramble the cube

    env.screen_width = 600  # you can reduce the screen size if FPS are too low
    env.screen_height = 600

    env.cap_fps = 10  # env assumes that you are close to this fps (controls might be weird if it's too far away from it)
    env.rotation_step = 5  # higher values makes the animation of rotating take more frames and vice versa, =90 rotations aren't animated

    model = RandomAgent()

    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)

    env.close()
