import pygame
import numpy as np
import time
import objects3D
from render import ProjectionRenderer, RaycastRenderer
from functions import *

if __name__ == "__main__":
    WIDTH, HEIGHT = 800, 800
    p = ProjectionRenderer(100, WIDTH, HEIGHT)
    colours = [
        (255, 213, 0),
        (0, 155, 72),
        (200, 100, 0),
        (185, 0, 0),
        (0, 69, 173),
        (255, 255, 255),
    ]
    objects_to_render = []
    for x in [100, 0, -100]:
        for y in [100, 0, -100]:
            for z in [100, 0, -100]:
                objects_to_render.append(
                    objects3D.Cube(np.array([x, y, z]), 90, colours)
                )
    # cube1 = objects3D.Cube(np.array([100, 100, 0]), 80, colours)
    # cube2 = objects3D.Cube(np.array([100, -100, 0]), 80, colours)
    # objects_to_render = [cube1, cube2]
    pygame.display.set_caption("Rubicks Cube")
    sphere = Sphere(800)
    look_point = np.array([0, 0, 0])

    over = False
    delta_theta, delta_phi = 0, 0
    time_passed = None
    myFont = pygame.font.SysFont("Comic Sans MS", 30)
    while not over:
        cur_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = True
            # key pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    delta_theta = -1
                elif event.key == pygame.K_DOWN:
                    delta_theta = 1
                elif event.key == pygame.K_RIGHT:
                    delta_phi = 1
                elif event.key == pygame.K_LEFT:
                    delta_phi = -1
            # key released
            elif event.type == pygame.KEYUP:
                if (event.key == pygame.K_UP and delta_theta == -1) or (
                    event.key == pygame.K_DOWN and delta_theta == 1
                ):
                    delta_theta = 0
                elif (event.key == pygame.K_RIGHT and delta_phi == 1) or (
                    event.key == pygame.K_LEFT and delta_phi == -1
                ):
                    delta_phi = 0
            elif event.type == pygame.MOUSEWHEEL:
                sphere.radius -= event.y * 20

        pov = sphere.rotate(delta_theta, delta_phi)

        p.render(pov, look_point, objects_to_render)
        if time_passed != None:
            text_surface = myFont.render(
                "FPS {}".format(int(1 / time_passed)), False, (255, 255, 255)
            )
            p.dis.blit(text_surface, (10, 10))
            pygame.display.update()

        time_passed = time.time() - cur_time
        wait = 1 / 60 - time_passed
        # 60 fps
        if wait > 0:
            time.sleep(wait)
