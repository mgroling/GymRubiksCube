# solve the Rubik's Cube by using a shortest-path algorithm
import gym
import gym_rubiks_cube
import numpy as np

from gym_rubiks_cube.envs.rubiksCubeEnv import TransformCubeObject


def reconstruct_path(v, visited_nodes):
    cur = v
    actions = []
    temp = visited_nodes[cur.tobytes()]
    while temp[1] != None:
        actions.append(temp[1])
        cur = temp[0]
        temp = visited_nodes[cur.tobytes()]

    actions.reverse()

    return actions


def solveCubeBFS(initial_state):
    transform = TransformCubeObject()
    visited_nodes = {initial_state.tobytes(): (None, None)}
    queue = []

    queue.append(initial_state)
    j = 0
    while len(queue) > 0:
        v = queue[0]
        if transform.isSolved(v):
            return reconstruct_path(v, visited_nodes)

        for i in range(18):
            w = transform(v, i)
            if not w.tobytes() in visited_nodes:
                queue.append(w)
                visited_nodes[w.tobytes()] = (v, i)
        j += 1
        del queue[0]
        if j % 10000 == 0:
            print("timestep {} finished".format(j))

    return "No solution found"


if __name__ == "__main__":
    env = gym.make("RubiksCube-v0")

    env.scramble_params = 4

    obs = env.reset()

    actions = solveCubeBFS(obs)

    transform = TransformCubeObject()

    print("Number of actions to solve the Rubik's Cube:", len(actions))

    state = obs

    for action in actions:
        env.render()
        env.step(action)

    while True:
        env.render()
