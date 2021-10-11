# solve the Rubik's Cube by using a shortest-path algorithm
import gym
import gym_rubiks_cube

if __name__ == "__main__":
    env = gym.make("RubiksCube-v0")

    env.reset()

    while True:
        env.render()
