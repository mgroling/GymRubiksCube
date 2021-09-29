from logging import error
import numpy as np


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

    def convertBarycentricCoordinates(self, x, y):
        return self.origin + x * self.vector1 + y * self.vector2


class Sphere:
    """https://math.stackexchange.com/questions/268064/move-a-point-up-and-down-along-a-sphere"""

    def __init__(self, radius) -> None:
        self.radius = radius
        self.current_pos = [90, 0]

    def rotate(self, x, y):
        self.current_pos[0] = (self.current_pos[0] + x) % 360
        self.current_pos[1] = (self.current_pos[1] + y) % 360

        theta, phi = np.deg2rad(self.current_pos[0]), np.deg2rad(self.current_pos[1])
        return np.array(
            [
                self.radius * np.sin(theta) * np.cos(phi),
                self.radius * np.sin(theta) * np.sin(phi),
                self.radius * np.cos(theta),
            ]
        )


class RotationMatrix3D:
    def __init__(self) -> None:
        pass

    def __call__(
        self, object_to_rotate: np.ndarray, axis: int, angle: float
    ) -> np.ndarray:
        assert (
            len(object_to_rotate.shape) == 2 and object_to_rotate.shape[0] == 3
        ), "Invalid shape of object to rotate, must be of shape (3, n)"
        if axis == 0:
            rotation_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            )
        elif axis == 1:
            rotation_matrix = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )
        elif axis == 2:
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise error("Invalid argument for axis, options are 0, 1, 2")

        return np.matmul(rotation_matrix, object_to_rotate)


def getQuadrant(x: float, y: float):
    if x == 0 and y == 0:
        return -1
    if x >= 0 and y >= 0:
        return 1
    elif x > 0 and y < 0:
        return 2
    elif x <= 0 and y <= 0:
        return 3
    elif x < 0 and y > 0:
        return 4
