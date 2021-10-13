# Gym Rubiks Cube

This is an [OpenAi gym environment](https://gym.openai.com/) for the popular combination puzzle of the [Rubik's Cube](https://en.wikipedia.org/wiki/Rubik%27s_Cube).

![RubiksCubeAnimation](/img/rubiksCubeAnimation.gif)

## Installation and Usage

    git clone https://github.com/marc131183/GymRubiksCube.git
    cd GymRubiksCube
    pip3 install -e .
    
### Example Usage
    
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
  
Example Usage can also be found [here](https://github.com/marc131183/GymRubiksCube/blob/main/experiments/example.py).

## Action and Observation Space

Action space: Discrete(18)

Observation space: (54,) np.ndarray with values of 0 to 5 representing the colors

## Content

- Gym environment for the Rubik's Cube (3x3x3)
- Visualization of actions with rendering of a virtual Rubik's Cube
- Visualization also offers the option to view the Cube from different perspectives (via arrow-keys/wasd) + zoom with mousewheel
- (soon) Algorithms that solve the Rubik's Cube by using the env
- Render engine that supports all 3d objects consisting of triangles (although the number of triangles should be kept low to keep a decent number of frames per second (doesn't use GPU))
