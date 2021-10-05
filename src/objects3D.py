import numpy as np
import abc
from typing import List


class Triangle3D:
    def __init__(
        self,
        origin: np.ndarray,
        vec1: np.ndarray,
        vec2: np.ndarray,
        fill_color,
    ) -> None:
        self.origin = origin
        self.vec1 = vec1
        self.vec2 = vec2
        self.fill_color = fill_color

    def __call__(self, u, v) -> np.ndarray:
        return self.origin + u * self.vec1 + v * self.vec2


def convertRectangleToTriangles(
    rect_origin: np.ndarray, rect_vec1: np.ndarray, rect_vec2: np.ndarray, fill_color
) -> List[Triangle3D]:
    return [
        Triangle3D(
            rect_origin,
            rect_vec1,
            rect_vec2,
            fill_color,
        ),
        Triangle3D(
            rect_origin + rect_vec1 + rect_vec2,
            -rect_vec1,
            -rect_vec2,
            fill_color,
        ),
    ]


class Renderable(abc.ABC):
    @abc.abstractmethod
    def get_triangles(self) -> List[Triangle3D]:
        pass


class Cube(Renderable):
    def __init__(self, center: np.ndarray, size: float, color) -> None:
        vertices = np.array(
            [
                [center[0] - size / 2, center[1] - size / 2, center[2] - size / 2],
                [center[0] + size / 2, center[1] - size / 2, center[2] - size / 2],
                [center[0] - size / 2, center[1] + size / 2, center[2] - size / 2],
                [center[0] + size / 2, center[1] + size / 2, center[2] - size / 2],
                [center[0] - size / 2, center[1] - size / 2, center[2] + size / 2],
                [center[0] + size / 2, center[1] - size / 2, center[2] + size / 2],
                [center[0] - size / 2, center[1] + size / 2, center[2] + size / 2],
                [center[0] + size / 2, center[1] + size / 2, center[2] + size / 2],
            ]
        )

        face_groups = [(0, 1, 2), (0, 1, 4), (0, 2, 4), (7, 6, 3), (7, 5, 3), (7, 6, 5)]
        self.triangles = []
        for i, group in enumerate(face_groups):
            col = color if type(color) != list else color[i]
            origin, vec1, vec2 = (
                vertices[group[0]],
                vertices[group[1]] - vertices[group[0]],
                vertices[group[2]] - vertices[group[0]],
            )
            self.triangles.extend(
                convertRectangleToTriangles(
                    origin,
                    vec1,
                    vec2,
                    col,
                )
            )

    def get_triangles(self) -> List[Triangle3D]:
        return self.triangles


class Pyramid(Renderable):
    def __init__(
        self,
        ground_origin: np.ndarray,
        ground_vec1: np.ndarray,
        ground_vec2: np.ndarray,
        top: np.ndarray,
        color,
    ) -> None:
        self.triangles = []
        ground_tri1 = Triangle3D(ground_origin, ground_vec1, ground_vec2, color)
        ground_tri2 = Triangle3D(
            ground_origin + ground_vec1 + ground_vec2, -ground_vec1, -ground_vec2, color
        )
        self.triangles = [ground_tri1, ground_tri2]
        ground_vertices = [
            ground_tri1(0, 0),
            ground_tri1(0, 1),
            ground_tri1(1, 0),
            ground_tri1(1, 1),
        ]

        triangle_groups = [(0, 1), (0, 2), (3, 1), (3, 2)]
        for group in triangle_groups:
            self.triangles.append(
                Triangle3D(
                    ground_vertices[group[0]],
                    ground_vertices[group[1]] - ground_vertices[group[0]],
                    top - ground_vertices[group[0]],
                    color,
                )
            )

    def get_triangles(self) -> List[Triangle3D]:
        return self.triangles


class TestClass(Renderable):
    def __init__(self, origin, vec1, vec2) -> None:
        self.triangles = convertRectangleToTriangles(origin, vec1, vec2, [255, 0, 0])

    def get_triangles(self) -> List[Triangle3D]:
        return self.triangles
