import numpy as np
import pygame
import objects3D
from functions import *


class ProjectionRenderer:
    def __init__(self, canvas_distance: float, width: int, height: int) -> None:
        super().__init__()
        self.switch = True
        self.last_quadrant = None
        self.canvas_distance = canvas_distance
        pygame.init()
        self.width, self.height = width, height
        self.dis = pygame.display.set_mode((width, height))

    def getCanvasCoordinates(self, pov: np.array, look_point: np.array, objects: list):
        # create a plane that is perpendicular to the view vector and use it as canvas
        u = look_point - pov
        cur_quadrant = getQuadrant(u[0], u[1])
        if self.last_quadrant != None and abs(cur_quadrant - self.last_quadrant) > 1:
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

    def computeVisibleFaces(
        self, objects: list, canvas_coords, num_vertices_total: int
    ):
        # reset render info from previous render
        for obj in objects:
            for face in obj.faces:
                face._resetCoordsInfo()

        self.dis.fill((0, 0, 0))
        visible_faces = []
        for _ in range(num_vertices_total):
            object_with_min_vertex = min(canvas_coords, key=lambda x: x[0][1])
            index_of_object = canvas_coords.index(object_with_min_vertex)
            min_vertex = object_with_min_vertex[0]

            if not isAnyFaceOverVertex(visible_faces, min_vertex[0]):
                for face in objects[index_of_object].faces:
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

            del object_with_min_vertex[0]

        return visible_faces

    def render(self, pov: np.array, look_point: np.array, objects: list):
        canvas_coordinates, num_vertices_total = self.getCanvasCoordinates(
            pov, look_point, objects
        )
        visible_faces = self.computeVisibleFaces(
            objects, canvas_coordinates, num_vertices_total
        )

        visible_faces.sort(key=lambda x: len(x.getPolygonOfSelf()))
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
    p = ProjectionRenderer(100, 800, 800)
    colours = [
        (255, 213, 0),
        (0, 155, 72),
        (200, 100, 0),
        (185, 0, 0),
        (0, 69, 173),
        (255, 255, 255),
    ]
    cube = objects3D.Cube(np.array([0, 0, 0]), 300, colours)

    over = False
    first = True
    while not over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                over = True

        if first:
            p.render(np.array([-800, -800, 0]), np.array([0, 0, 0]), [cube])
            first = False
