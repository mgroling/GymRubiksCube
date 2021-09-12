from abc import get_cache_token

from numpy.lib.polynomial import poly
from main import Cube
import numpy as np
import pygame
import objects3D


def getQuadrant(x: float, y: float):
    if x >= 0 and y >= 0:
        return 1
    elif x > 0 and y < 0:
        return 2
    elif x <= 0 and y <= 0:
        return 3
    elif x < 0 and y < 0:
        return 4


def doesLineSegmentIntersectRayFromPoint(
    point: np.array,
    ray_direction: np.array,
    lineSegment_start: np.array,
    lineSegment_end: np.array,
):
    # compute intersection of them and check if it's between line start and end
    # check if ray and line are parallel (if A is invertible)
    A = np.array([(lineSegment_end - lineSegment_start), -ray_direction]).T
    if np.linalg.det(A) == 0:
        return False
    b = point - lineSegment_start
    sol = np.linalg.solve(A, b)
    if sol[0] >= 0 and sol[0] <= 1 and sol[1] >= 0:
        return True
    else:
        return False


def isVertexInsidePolygon2D(vertex: np.array, lines: list):
    # if ray intersects an even number of times, then vertex is outside of polygon
    intersections = 0
    for line in lines:
        print("current line", line)
        if doesLineSegmentIntersectRayFromPoint(
            vertex, np.array([1, 1]), line[0], line[1]
        ):
            intersections += 1
            print("intersects!")

    # in theory, we'd have to subtract 1 if the ray goes through an exact point per point, but it's very unlikely so we can ignore it for now

    print("num intersections", intersections)
    if intersections % 2 == 0:
        return False
    else:
        return True


def isAnyFaceOverVertex(faces: list, vertex: np.array):
    for face in faces:
        # create the lines between visible vertices of a given face (such that they are in order as given by the face)
        visible_vertices_ordered = face.getPolygonOfSelf()
        visible_lines = [
            (visible_vertices_ordered[i], visible_vertices_ordered[i + 1])
            for i in range(len(visible_vertices_ordered) - 1)
        ] + [(visible_vertices_ordered[-1], visible_vertices_ordered[0])]
        if isVertexInsidePolygon2D(vertex, visible_lines):
            return True

    return False


class Plane:
    def __init__(self, origin: np.array, vector1: np.array, vector2: np.array) -> None:
        self.origin = origin
        self.vector1 = vector1
        self.vector2 = vector2

    def getBarycentricCoordinates(self, point: np.array, direction: np.array):
        a = np.array([self.vector1, self.vector2, -direction]).T
        b = point - self.origin
        sol = np.linalg.solve(a, b)

        return np.array([sol[0], sol[1]])


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

            print("current vertex", min_vertex[0])
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
            p.render(np.array([-800, -800, 400]), np.array([0, 0, 0]), [cube])
            first = False
