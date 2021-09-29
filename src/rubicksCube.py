import pygame
import numpy as np
import time
from objects3D import Cube
from render import Scene
from functions import Sphere
from PIL import Image


class RubicksCube:
    def __init__(self, setup_render=False, screen_width=600, screen_height=600) -> None:
        self.scene = None
        # define structure of cube, by remembering indices of objects, if the flattened array is in ascending order the cube is solved
        self.structure = np.arange(27).reshape(3, 3, 3)

        # define scene for rendering
        colors = [
            (255, 213, 0),
            (0, 155, 72),
            (200, 100, 0),
            (0, 69, 173),
            (185, 0, 0),
            (255, 255, 255),
        ]
        objects_to_render = []
        for x in [100, 0, -100]:
            for y in [100, 0, -100]:
                for z in [100, 0, -100]:
                    col = [(0, 0, 0) for i in range(6)]
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


if __name__ == "__main__":
    r = RubicksCube()
