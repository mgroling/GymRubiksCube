import pygame
import numpy as np
import time
import gym
import sys
from objects3D import Cube
from render import Scene
from functions import Sphere
from PIL import Image


class RubicksCubeEnv(gym.Env):
    def __init__(self) -> None:
        # variables for rendering
        self.__setup_render = False
        self._scene = None
        self._sphere = None
        self._dis = None
        self._font = None
        self._look_point = None
        self._screen_width = None
        self._screen_height = None
        self._delta_theta, self._delta_phi = None, None

        self.cap_fps = 10

        # TODO: define action and observation space
        self.action_space = gym.spaces.Discrete(18)
        self.reset()

    def reset(self) -> np.ndarray:
        # structure: axis = 0 is top layer, = 1 is middle layer, = 2 is top layer
        # 0 1 2
        # 3 4 5
        # 6 7 8
        self.structure = np.arange(27).reshape(3, 3, 3)

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
        self._screen_width, self._screen_height = 600, 600
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


if __name__ == "__main__":
    env = RubicksCubeEnv()

    i = 0
    while True:
        env.render()
        env.step(np.random.randint(18))
        # env.step(i)
        i = (i + 1) % 18
