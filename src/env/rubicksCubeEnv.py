import pygame
import numpy as np
import time
import gym
import sys
from PIL import Image
from objects3D import Cube
from render import Scene
from functions import Sphere


class TransformeCubeObject:
    def __init__(self) -> None:
        # fmt: off
        self.transformation_permutations = [np.arange(54) for _ in range(18)]

        # action = 0: top-layer counter-clockwise rotation
        self.transformation_permutations[0][
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]
        ] = [2, 5, 8, 1, 4, 7, 0, 3, 6, 36, 37, 38, 9, 10, 11, 18, 19, 20, 27, 28, 29]
        # action = 1: top-layer clockwise rotation
        self.transformation_permutations[1][
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38]
        ] = [6, 3, 0, 7, 4, 1, 8, 5, 2, 18, 19, 20, 27, 28, 29, 36, 37, 38, 9, 10, 11]
        # action = 2: middle-layer horizontal counter-clockwise rotation
        self.transformation_permutations[2][
            [12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41]
        ] = [39, 40, 41, 12, 13, 14, 21, 22, 23, 30, 31, 32]
        # action = 3: middle-layer horizontal clockwise rotation
        self.transformation_permutations[3][
            [12, 13, 14, 21, 22, 23, 30, 31, 32, 39, 40, 41]
        ] = [21, 22, 23, 30, 31, 32, 39, 40, 41, 12, 13, 14]
        # action = 4: bottom-layer counter-clockwise rotation
        self.transformation_permutations[4][
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44]
        ] = [47, 50, 53, 46, 49, 52, 45, 48, 51, 42, 43, 44, 15, 16, 17, 24, 25, 26, 33, 34, 35]
        # action = 5: bottom-layer clockwise rotation
        self.transformation_permutations[5][
            [45, 46, 47, 48, 49, 50, 51, 52, 53, 15, 16, 17, 24, 25, 26, 33, 34, 35, 42, 43, 44]
        ] = [51, 48, 45, 52, 49, 46, 53, 50, 47, 24, 25, 26, 33, 34, 35, 42, 43, 44, 15, 16, 17]
        # action = 6: right-layer downwards rotation
        self.transformation_permutations[6][
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 5, 8, 11, 14, 17, 27, 30, 33, 47, 50, 53]
        ] = [20, 23, 26, 19, 22, 25, 18, 21, 24, 33, 30, 27, 2, 5, 8, 47, 50, 53, 17, 14, 11]
        # action = 7: right-layer upwards rotation
        self.transformation_permutations[7][
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 2, 5, 8, 11, 14, 17, 27, 30, 33, 47, 50, 53]
        ] = [24, 21, 18, 25, 22, 19, 26, 23, 20, 11, 14, 17, 53, 50, 47, 8, 5, 2, 27, 30, 33]
        # action = 8: middle-layer vertical downwards rotation
        self.transformation_permutations[8][
            [1, 4, 7, 10, 13, 16, 28, 31, 34, 46, 49, 52]
        ] = [34, 31, 28, 1, 4, 7, 46, 49, 52, 16, 13, 10]
        # action = 9: middle-layer vertical upwards rotation
        self.transformation_permutations[9][
            [1, 4, 7, 10, 13, 16, 28, 31, 34, 46, 49, 52]
        ] = [10, 13, 16, 52, 49, 46, 7, 4, 1, 28, 31, 34]
        # action = 10: left-layer downwards rotation
        self.transformation_permutations[10][
            [36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 3, 6, 9, 12, 15, 29, 32, 35, 45, 48, 51]
        ] = [42, 39, 36, 43, 40, 37, 44, 41, 38, 35, 32, 29, 0, 3, 6, 45, 48, 51, 15, 12, 9]
        # action = 11: left-layer upwards rotation
        self.transformation_permutations[11][
            [36, 37, 38, 39, 40, 41, 42, 43, 44, 0, 3, 6, 9, 12, 15, 29, 32, 35, 45, 48, 51]
        ] = [38, 41, 44, 37, 40, 43, 36, 39, 42, 9, 12, 15, 51, 48, 45, 6, 3, 0, 29, 32, 35]
        # action = 12: back-layer counter-clockwise rotation
        self.transformation_permutations[12][
            [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 20, 23, 26, 36, 39, 42, 45, 46, 47]
        ] = [33, 30, 27, 34, 31, 28, 35, 32, 29, 20, 23, 26, 47, 46, 45, 2, 1, 0, 36, 39, 42]
        # action = 13: back-layer clockwise rotation
        self.transformation_permutations[13][
            [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 20, 23, 26, 36, 39, 42, 45, 46, 47]
        ] = [29, 32, 35, 28, 31, 34, 27, 30, 33, 42, 39, 36, 0, 1, 2, 45, 46, 47, 26, 23, 20]
        # action = 14: middle-layer counter-clockwise rotation
        self.transformation_permutations[14][
            [3, 4, 5, 19, 22, 25, 37, 40, 43, 48, 49, 50]
        ] = [19, 22, 25, 50, 49, 48, 5, 4, 3, 37, 40, 43]
        # action = 15: middle-layer clockwise rotation
        self.transformation_permutations[15][
            [3, 4, 5, 19, 22, 25, 37, 40, 43, 48, 49, 50]
        ] = [43, 40, 37, 3, 4, 5, 48, 49, 50, 25, 22, 19]
        # action = 16: front-layer counter-clockwise rotation
        self.transformation_permutations[16][
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 18, 21, 24, 38, 41, 44, 51, 52, 53]
        ] = [11, 14, 17, 10, 13, 16, 9, 12, 15, 18, 21, 24, 53, 52, 51, 8, 7, 6, 38, 41, 44]
        # action = 17: front-layer clockwise rotation
        self.transformation_permutations[17][
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 6, 7, 8, 18, 21, 24, 38, 41, 44, 51, 52, 53]
        ] = [15, 12, 9, 16, 13, 10, 17, 14, 11, 44, 41, 38, 6, 7, 8, 51, 52, 53, 24, 21, 18]

        # fmt: on

    def __call__(self, current_state: np.ndarray, action: int) -> np.ndarray:
        return np.array(current_state)[self.transformation_permutations[action]]


class RubicksCubeEnv(gym.Env):
    def __init__(self) -> None:
        # variables for rendering
        self.__setup_render = False
        self._scene = None
        self._sphere = None
        self._dis = None
        self._font = None
        self._look_point = None
        self._screen_width = 600
        self._screen_height = 600
        self._delta_theta, self._delta_phi = None, None

        self.cap_fps = 10

        # TODO: define action and observation space
        self.action_space = gym.spaces.Discrete(18)
        self.reset()

    def reset(self) -> np.ndarray:
        # structure: row = 0 is top layer, row = 1 is middle layer, row = 2 is top layer
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.structure = np.arange(27).reshape(3, 3, 3)
        # define a vector representing the color of each side
        # number from 0-5 are mapped to the colors in the following order white, red, blue, orange, green and yellow
        self.color_vector = np.array(
            [[j for _ in range(9)] for j in range(6)]
        ).flatten()

    def step(self, action: int) -> np.ndarray:
        # NOT AS GOOD; BETTER USE THIS: VECTOR REPRESENTATION (page 4)
        # https://dl.acm.org/doi/pdf/10.1145/800058.801107?casa_token=Qv5syAkQaZAAAAAA:JmbMhpHxjpeC8Oijx1_au6y-nNxiTfh9lfS6B06kUZQowLNixqXEqTxgNwOcKeTwP-oswhlvwH8
        if self.__setup_render:
            axis = action // 6
            index = action % 6 // 2
            # rotate either counter-clockwise or clockwise (if action % 2 == 0 -> counter-clockwise, else clockwise)
            flip_axis = action % 2
            step_rot = 5 if action % 2 == 0 else -5

            # show animation of doing a rotation
            for _ in range(0, 90, abs(step_rot)):
                if axis == 0:
                    self._scene.rotateObjects(
                        self.structure[index].flatten(), 2, np.deg2rad(step_rot)
                    )
                elif axis == 1:
                    self._scene.rotateObjects(
                        self.structure[:, index].flatten(), 1, np.deg2rad(step_rot)
                    )
                else:
                    self._scene.rotateObjects(
                        self.structure[:, :, index].flatten(), 0, np.deg2rad(step_rot)
                    )
                self.render()

            # update structure
            if axis == 0:
                self.structure[index] = np.flip(self.structure[index].T, axis=flip_axis)
            elif axis == 1:
                self.structure[:, index] = np.flip(
                    self.structure[:, index].T, axis=1 - flip_axis
                )
            else:
                self.structure[:, :, index] = np.flip(
                    self.structure[:, :, index].T, axis=1 - flip_axis
                )

    def _setup_render(self) -> None:
        # define scene for rendering
        # TODO: use current state of cube for rendering
        self.reset()
        colors = [
            (255, 213, 0),
            (0, 155, 72),
            (200, 100, 0),
            (0, 69, 173),
            (185, 0, 0),
            (255, 255, 255),
        ]
        objects_to_render = []
        for z in [100, 0, -100]:
            for y in [100, 0, -100]:
                for x in [-100, 0, 100]:
                    col = [(0, 0, 0) for _ in range(6)]
                    if x < 0:
                        col[2] = colors[2]
                    elif x > 0:
                        col[4] = colors[4]
                    if y < 0:
                        col[1] = colors[1]
                    elif y > 0:
                        col[3] = colors[3]
                    if z < 0:
                        col[0] = colors[0]
                    elif z > 0:
                        col[5] = colors[5]
                    objects_to_render.append(Cube(np.array([x, y, z]), 93, col))

        self._scene = Scene(
            self._screen_width // 2,
            self._screen_height // 2,
            objects_to_render,
            400,
            (50, 50, 50),
        )
        self._sphere = Sphere(800)
        self._look_point = np.array([0, 0, 0])

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Rubik's Cube")
        self._dis = pygame.display.set_mode((self._screen_width, self._screen_height))
        self._font = pygame.font.SysFont("Comic Sans MS", 30)
        self._delta_theta, self._delta_phi = 0, 0

    def render(self) -> None:
        if not self.__setup_render:
            self._setup_render()
            self.__setup_render = True

        cur_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # key pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self._delta_theta = -3
                elif event.key == pygame.K_DOWN:
                    self._delta_theta = 3
                elif event.key == pygame.K_RIGHT:
                    self._delta_phi = 3
                elif event.key == pygame.K_LEFT:
                    self._delta_phi = -3
            # key released
            elif event.type == pygame.KEYUP:
                if (event.key == pygame.K_UP and self._delta_theta < 0) or (
                    event.key == pygame.K_DOWN and self._delta_theta > 0
                ):
                    self._delta_theta = 0
                elif (event.key == pygame.K_RIGHT and self._delta_phi > 0) or (
                    event.key == pygame.K_LEFT and self._delta_phi < 0
                ):
                    self._delta_phi = 0
            elif event.type == pygame.MOUSEWHEEL:
                self._sphere.radius -= event.y * 20

        pov = self._sphere.rotate(self._delta_theta, self._delta_phi)
        color_map = self._scene.render(pov, self._look_point)
        img = Image.fromarray(color_map, mode="RGB")
        img = img.resize(
            (self._screen_width, self._screen_height), resample=Image.NEAREST
        )
        img = np.array(img)
        pygame.surfarray.blit_array(self._dis, img)

        time_passed = time.time() - cur_time
        text_surface = self._font.render(
            "FPS {}".format(min(int(1 / time_passed), self.cap_fps)),
            False,
            (255, 255, 255),
        )
        self._dis.blit(text_surface, (10, 10))

        pygame.display.update()

        wait = 1 / self.cap_fps - time_passed
        if wait > 0:
            time.sleep(wait)

    @property
    def screen_width(self) -> int:
        return self._screen_width

    @screen_width.setter
    def screen_width(self, value: int):
        assert (
            type(value) == int and value > 1
        ), "screen width must be an integer and bigger than one"
        self._screen_width = value
        self.__setup_render = False

    @property
    def screen_height(self) -> int:
        return self._screen_height

    @screen_height.setter
    def screen_height(self, value: int):
        assert (
            type(value) == int and value > 1
        ), "screen height must be an integer and bigger than one"
        self._screen_height = value
        self.__setup_render = False


if __name__ == "__main__":
    env = RubicksCubeEnv()

    # temp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # for i in range(6):
    #     print(temp + i * 9)

    # temp = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    # print(np.flip(temp.T + 9, axis=1))

    transformer = TransformeCubeObject()
    color_vector = np.array([[j for _ in range(9)] for j in range(6)]).flatten()
    pair = 16, 17
    temp = transformer(transformer(color_vector, pair[0]), pair[1])
    print(np.array(temp == sorted(temp)).all())
    temp = transformer(transformer(color_vector, pair[1]), pair[0])
    print(np.array(temp == sorted(temp)).all())

    # i = 0
    # while True:
    #     env.render()
    #     # env.step(np.random.randint(18))
    #     env.step(12)
    #     i = (i + 1) % 18
