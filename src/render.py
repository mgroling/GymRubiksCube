import numpy as np
import pygame
import time
import timeit
import objects3D
from functions import *


class ProjectionRenderer:
    def __init__(self, canvas_distance: float, width: int, height: int) -> None:
        super().__init__()
        self.switch = True
        self.last_quadrant = None
        self.canvas_distance = canvas_distance
        pygame.init()
        pygame.font.init()
        self.width, self.height = width, height
        self.dis = pygame.display.set_mode((width, height))

    # TODO: there's still a problem when u[0] and u[1] are zero, not sure how to fix it yet
    def getCanvasCoordinates(self, pov: np.array, look_point: np.array, objects: list):
        # create a plane that is perpendicular to the view vector and use it as canvas
        u = look_point - pov
        cur_quadrant = getQuadrant(u[0], u[1])
        if cur_quadrant != -1:
            if (
                self.last_quadrant != None
                and max(cur_quadrant, self.last_quadrant)
                - min(cur_quadrant, self.last_quadrant)
                == 2
            ):
                self.switch = not self.switch
            self.last_quadrant = cur_quadrant

        if self.switch:
            v = np.array([u[1], -u[0], 0])
        else:
            v = np.array([-u[1], u[0], 0])
        w = np.cross(u, v)

        u, v, w = u / np.linalg.norm(u), v / np.linalg.norm(v), w / np.linalg.norm(w)
        canvas = Plane(-u / 2, v, w)

        num_vertices_total = 0
        canvas_coords = []
        for obj in objects:
            canvas_coords.append([])
            for i, vertex in enumerate(obj.vertices):
                canvas_coords[-1].append(
                    (
                        canvas.getBarycentricCoordinates(vertex, pov - vertex),
                        np.linalg.norm(pov - vertex),
                        i,
                    )
                )
                num_vertices_total += 1
            canvas_coords[-1].sort(key=lambda x: x[1])

        return canvas_coords, num_vertices_total

    # this function is terribly slow, need to find a better/correct way to determine visiblity of faces
    def computeVisibleFaces(
        self, objects: list, canvas_coords, num_vertices_total: int
    ):
        # reset render info from previous render
        for obj in objects:
            for face in obj.faces:
                face._resetCoordsInfo()

        self.dis.fill((50, 50, 50))
        visible_faces = []
        min_distances = [elem[0][1] for elem in canvas_coords]
        current_elem = [0 for elem in canvas_coords]
        for _ in range(num_vertices_total):
            index_of_object_with_min_vertex = min_distances.index(min(min_distances))
            min_vertex = canvas_coords[index_of_object_with_min_vertex][
                current_elem[index_of_object_with_min_vertex]
            ]

            if not isAnyFaceOverVertex(visible_faces, min_vertex[0]):
                for face in objects[index_of_object_with_min_vertex].faces:
                    # set the vertex to visible in each face that contains it
                    if min_vertex[2] in face.vertices:
                        face._coordsInfo[
                            face.vertices.index(min_vertex[2])
                        ] = min_vertex
                    # if new visible face (>= 3 visible vertices), add it to visible_faces
                    if (
                        sum([1 for elem in face._coordsInfo if elem != None]) >= 3
                        and not face in visible_faces
                    ):
                        visible_faces.append(face)

            current_elem[index_of_object_with_min_vertex] += 1
            if (
                current_elem[index_of_object_with_min_vertex]
                > len(canvas_coords[index_of_object_with_min_vertex]) - 1
            ):
                min_distances[index_of_object_with_min_vertex] = np.inf
            else:
                min_distances[index_of_object_with_min_vertex] = canvas_coords[
                    index_of_object_with_min_vertex
                ][current_elem[index_of_object_with_min_vertex]][1]

        return visible_faces

    def render(self, pov: np.array, look_point: np.array, objects: list):
        cur_time = time.time()
        canvas_coordinates, num_vertices_total = self.getCanvasCoordinates(
            pov, look_point, objects
        )
        time_passed = time.time() - cur_time
        cur_time = time.time()
        visible_faces = self.computeVisibleFaces(
            objects, canvas_coordinates, num_vertices_total
        )
        time_passed = time.time() - cur_time

        visible_faces = [
            elem for elem in visible_faces if len(elem.getPolygonOfSelf()) == 4
        ]
        # render faces
        for face in visible_faces:
            polygon = face.getPolygonOfSelf()
            pygame.draw.polygon(
                self.dis,
                face.colour,
                np.array([self.width / 2, self.height / 2]) + polygon,
            )
            # for vertex in polygon:
            #     pygame.draw.rect(
            #         self.dis,
            #         (255, 0, 0),
            #         [self.width / 2 + vertex[0], self.height / 2 + vertex[1], 5, 5],
            #     )
        pygame.display.update()


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
        wait = 0.0166666666 - time_passed
        # 60 fps
        if wait > 0:
            time.sleep(wait)
