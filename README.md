# Gym Rubiks Cube

This is an [OpenAi gym environment](https://gym.openai.com/) for the popular combination puzzle of the [Rubik's Cube](https://en.wikipedia.org/wiki/Rubik%27s_Cube).

![RubiksCubeAnimation](/img/rubiksCubeAnimation.gif)

## Content

- Gym environment for the Rubik's Cube (3x3)
- Visualization of actions with rendering of a virtual Rubik's Cube
- Visualization also offers the option to view the Cube from different perspectives (via arrow-keys/wasd) + zoom with mousewheel
- (soon) Algorithms that solve the Rubik's Cube by using the env
- Render engine that supports all 3d objects consisting of triangles (although the number of triangles should be kept low to keep a decent number of frames per second (doesn't use GPU))
