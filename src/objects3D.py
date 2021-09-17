import numpy as np
import abc


class Vector3D(np.ndarray):
    def __new__(cls, input_array):
        assert len(input_array) == 3
        obj = np.asarray(input_array).view(cls)
        return obj


class Ray:
    def __init__(self, origin: Vector3D, direction: Vector3D) -> None:
        self.origin = origin
        self.direction = direction

    def __call__(self, t: float) -> Vector3D:
        return self.origin + t * self.direction


class HitRecord:
    def __init__(self) -> None:
        self.t = None
        self.color = None
        self.normal = None

    def __str__(self) -> str:
        return "t = {}, color = {}, normal = {}".format(self.t, self.color, self.normal)


class Hittable(abc.ABC):
    @abc.abstractmethod
    def hit(self, ray: Ray, hit_record: HitRecord):
        pass


class Rectangle3D(Hittable):
    def __init__(self, origin: Vector3D, vec1: Vector3D, vec2: Vector3D, color) -> None:
        self.origin = origin
        self.vec1 = vec1
        self.vec2 = vec2
        self.color = color

    def hit(self, ray: Ray, hit_record: HitRecord) -> bool:
        a = np.array([self.vec1, self.vec2, -ray.direction]).T
        if np.linalg.det(a) == 0:
            return False
        b = ray.origin - self.origin
        sol = np.linalg.solve(a, b)

        if sol[0] >= 0 and sol[0] <= 1 and sol[1] >= 0 and sol[1] <= 1 and sol[2] > 0:
            hit_record.t = sol[2]
            hit_record.color = self.color
            hit_record.normal = np.cross(self.vec1, self.vec2)
            hit_record.normal = hit_record.normal / np.linalg.norm(hit_record.normal)
            return True
        else:
            return False


class Cube(Hittable):
    def __init__(self, center: Vector3D, side_length: float, colors) -> None:
        vertices = [
            Vector3D(
                [
                    center[0] - side_length / 2,
                    center[1] - side_length / 2,
                    center[2] - side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] + side_length / 2,
                    center[1] - side_length / 2,
                    center[2] - side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] - side_length / 2,
                    center[1] + side_length / 2,
                    center[2] - side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] + side_length / 2,
                    center[1] + side_length / 2,
                    center[2] - side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] - side_length / 2,
                    center[1] - side_length / 2,
                    center[2] + side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] + side_length / 2,
                    center[1] - side_length / 2,
                    center[2] + side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] - side_length / 2,
                    center[1] + side_length / 2,
                    center[2] + side_length / 2,
                ]
            ),
            Vector3D(
                [
                    center[0] + side_length / 2,
                    center[1] + side_length / 2,
                    center[2] + side_length / 2,
                ]
            ),
        ]
        face_groups = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 2, 4),
            (7, 6, 5),
            (7, 6, 3),
            (7, 5, 3),
        ]
        self.rectangles = []
        for i, group in enumerate(face_groups):
            p1, p2, p3 = vertices[group[0]], vertices[group[1]], vertices[group[2]]
            origin, vec1, vec2 = p1, p2 - p1, p3 - p1
            self.rectangles.append(
                Rectangle3D(
                    origin, vec1, vec2, colors if type(colors) != list else colors[i]
                )
            )

    def hit(self, ray: Ray, hit_record: HitRecord) -> bool:
        any_hit = False
        for rect in self.rectangles:
            temp = HitRecord()
            if rect.hit(ray, temp):
                any_hit = True
                if hit_record.t == None or hit_record.t > temp.t:
                    hit_record.t = temp.t
                    hit_record.color = temp.color
                    hit_record.normal = temp.normal

        return any_hit


class Face:
    def __init__(self, vertices, color) -> None:
        # append the first vertex at the end again, such that we can create lines between them easily
        self.vertices = vertices
        self.color = color
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


class CubePrj:
    def __init__(
        self,
        center: np.array,
        size: float,
        color,
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
            col = color if type(color) != list else color[i]
            self.faces.append(Face(group, col))
