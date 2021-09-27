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
