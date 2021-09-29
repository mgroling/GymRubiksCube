import pygame
import numpy as np
import time
from objects3D import Cube, Pyramid
from render import Scene
from functions import Sphere
from PIL import Image

if __name__ == "__main__":
    WIDTH, HEIGHT = 300, 300
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
    # objects_to_render = [
    #     Pyramid(
    #         np.array([-100, -100, 0]),
    #         np.array([200, 0, 0]),
    #         np.array([0, 200, 0]),
    #         np.array([0, 0, 100]),
    #         (255, 0, 0),
    #     )
    # ]
    # objects_to_render = [Cube(np.array([0, 0, 0]), 100, colors)]

    scene = Scene(WIDTH, HEIGHT, objects_to_render, 400, (50, 50, 50))

    pygame.display.set_caption("Rubik's Cube")
    sphere = Sphere(800)
    look_point = np.array([0, 0, 0])

    cap_fps = 12
    over = False
    delta_theta, delta_phi = 0, 0
    time_passed = None
    pygame.init()
    pygame.font.init()
    dis = pygame.display.set_mode((2 * WIDTH, 2 * HEIGHT))
    myFont = pygame.font.SysFont("Comic Sans MS", 30)
    while not over:
        cur_time = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = True
            # key pressed
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    delta_theta = -3
                elif event.key == pygame.K_DOWN:
                    delta_theta = 3
                elif event.key == pygame.K_RIGHT:
                    delta_phi = 3
                elif event.key == pygame.K_LEFT:
                    delta_phi = -3
            # key released
            elif event.type == pygame.KEYUP:
                if (event.key == pygame.K_UP and delta_theta < 0) or (
                    event.key == pygame.K_DOWN and delta_theta > 0
                ):
                    delta_theta = 0
                elif (event.key == pygame.K_RIGHT and delta_phi > 0) or (
                    event.key == pygame.K_LEFT and delta_phi < 0
                ):
                    delta_phi = 0
            elif event.type == pygame.MOUSEWHEEL:
                sphere.radius -= event.y * 20

        pov = sphere.rotate(delta_theta, delta_phi)

        color_map = scene.render(pov, look_point)
        img = Image.fromarray(color_map, mode="RGB")
        img = img.resize((2 * WIDTH, 2 * HEIGHT), resample=Image.NEAREST)
        img = np.array(img)
        pygame.surfarray.blit_array(dis, img)
        # pygame.surfarray.blit_array(dis, color_map)
        if time_passed != None:
            text_surface = myFont.render(
                "FPS {}".format(min(int(1 / time_passed), cap_fps)),
                False,
                (255, 255, 255),
            )
            dis.blit(text_surface, (10, 10))

        pygame.display.update()

        time_passed = time.time() - cur_time
        wait = 1 / cap_fps - time_passed
        print(time_passed)
        if wait > 0:
            time.sleep(wait)
