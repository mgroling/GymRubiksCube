import numpy as np
import abc


class Ray:
    def __init__(self, origin: np.array, direction: np.array) -> None:
        self.origin = origin
        self.direction = direction

    def __call__(self, t: float) -> np.array:
        return self.origin + t * self.direction


class HitRecord:
    def __init__(self) -> None:
        self.t = None
        self.colour = None
        self.normal = None


class Hittable(abc.ABC):
    @abc.abstractmethod
    def hit(self, ray: Ray, hit_record: HitRecord) -> bool:
        pass


class Rectangle(Hittable):
    def __init__(self) -> None:
        super().__init__()


class Face:
    def __init__(self, vertices, colour) -> None:
        # append the first vertex at the end again, such that we can create lines between them easily
        self.vertices = vertices
        self.colour = colour
        self._coordsInfo = [None for i in range(len(self.vertices))]

    def _resetCoordsInfo(self):
        self._coordsInfo = [None for i in range(len(self.vertices))]

    def getPolygonOfSelf(self):
        visible_vertices_of_face = [
            (v[0], self.vertices.index(v[2])) for v in self._coordsInfo if v != None
        ]
        if len(visible_vertices_of_face) >= 3:
            visible_vertices_of_face.sort(key=lambda x: x[1])
            visible_vertices_of_face = np.array(
                [elem[0] for elem in visible_vertices_of_face]
            )
            return visible_vertices_of_face


class Cube:
    def __init__(
        self,
        center: np.array,
        size: float,
        colour,
    ) -> None:
        super().__init__()
        self.vertices = np.array(
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

        face_groups = [
            (0, 1, 3, 2),
            (0, 1, 5, 4),
            (0, 2, 6, 4),
            (1, 3, 7, 5),
            (2, 3, 7, 6),
            (4, 5, 7, 6),
        ]
        self.faces = []
        for i, group in enumerate(face_groups):
            col = colour if type(colour) != list else colour[i]
            self.faces.append(Face(group, col))
