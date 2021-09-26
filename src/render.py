import numpy as np
from functions import *
import objects3D as o3
from typing import List, Tuple


class Scene:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        objects_to_render: List[o3.Renderable],
        canvas_distance: float,
        bg_color,
    ) -> None:
        # save general paramters
        self.width = screen_width
        self.height = screen_height
        self.canvas_distance = canvas_distance
        self.bg_color = bg_color

        # save render parameters
        self.switch = True
        self.last_quadrant = None
        self.dis = None

        # set up internal structures for objects to render
        # for every object we save which triangles belong to it, in order to later modify objects easily
        self.objects = {i: [] for i, elem in enumerate(objects_to_render)}

        self.triangle_origins = []
        self.triangle_vec1s = []
        self.triangle_vec2s = []
        self.triangle_fill_colors = []

        index_tri = 0
        for i, object in enumerate(objects_to_render):
            for triangle in object.get_triangles():
                self.objects[i].append(index_tri)
                self.triangle_origins.append(triangle.origin)
                self.triangle_vec1s.append(triangle.vec1)
                self.triangle_vec2s.append(triangle.vec2)
                self.triangle_fill_colors.append(triangle.fill_color)
                index_tri += 1

        self.triangle_origins = np.array(self.triangle_origins)
        self.triangle_vec1s = np.array(self.triangle_vec1s)
        self.triangle_vec2s = np.array(self.triangle_vec2s)

    def _getCanvasVecs(
        self, pov: np.ndarray, look_point: np.ndarray
    ) -> Tuple[np.ndarray]:
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

        v, w = v / np.linalg.norm(v), w / np.linalg.norm(w)

        origin = pov + self.canvas_distance * (u / np.linalg.norm(u))
        # move origin to top left of canvas
        origin = origin - v * self.width / 2 - w * self.height / 2

        return origin, v, w

    def _renderRaycast(
        self,
        canvas_origin: np.ndarray,
        canvas_vecX: np.ndarray,
        canvas_vecY: np.ndarray,
    ):
        pass

    def _get2DCoordinates(
        self,
        pov: np.ndarray,
        canvas_origin: np.ndarray,
        canvas_vecX: np.ndarray,
        canvas_vecY: np.ndarray,
    ) -> np.ndarray:
        # create vertices for each rectangle/triangle in 3D
        tri_options = [(1, 0), (0, 1)]
        tri_vertices = np.empty((len(self.triangle_origins), 3, 3))
        tri_vertices[:, 0] = self.triangle_origins
        for i, option in enumerate(tri_options):
            tri_vertices[:, i + 1] = (
                self.triangle_origins
                + option[0] * self.triangle_vec1s
                + option[1] * self.triangle_vec2s
            )

        # now map these to 2D
        # create vector going from each vertex to the pov
        rays_to_pov_tri = pov[np.newaxis, np.newaxis] - tri_vertices
        # normalize them so we can use them as distance measure later
        rays_to_pov_tri = (
            rays_to_pov_tri / np.linalg.norm(rays_to_pov_tri, axis=2)[:, :, np.newaxis]
        )[:, :, :, np.newaxis]

        # now get the barycentric coordinates of each vertex on the plane (canvas)
        # add dimension for each vector to get it in matrix form
        canvas_vecX_repeated = np.repeat(canvas_vecX[np.newaxis], 4, axis=0)
        canvas_vecY_repeated = np.repeat(canvas_vecY[np.newaxis], 4, axis=0)
        canvas_vecX_repeated_tri = np.repeat(
            canvas_vecX_repeated[np.newaxis, :3], len(tri_vertices), axis=0
        )[:, :, :, np.newaxis]
        canvas_vecY_repeated_tri = np.repeat(
            canvas_vecY_repeated[np.newaxis, :3], len(tri_vertices), axis=0
        )[:, :, :, np.newaxis]

        # solve linear equation to get the 2D coordinates:
        # vertex + t * (pov - vertex) = canvas_origin + x * canvas_VecX + y * canvas_VecY
        matrix_tri = np.concatenate(
            [canvas_vecX_repeated_tri, canvas_vecY_repeated_tri, -rays_to_pov_tri],
            axis=3,
        )
        vector_tri = (tri_vertices - canvas_origin[np.newaxis, np.newaxis])[
            :, :, :, np.newaxis
        ]
        tri_x_y_t = np.matmul(np.linalg.inv(matrix_tri), vector_tri)

        return tri_x_y_t[:, :, :, 0]

    # TODO: parallelization of this should be easy, so DO IT!!!
    # assume v1.y > v2.y = v3.y
    def __rasterizeBottomFlatTriangle(self, v1, v2, v3):
        object_map = np.ones((self.width, self.height)) * np.inf
        slope1 = -(v2[0] - v1[0]) / (v2[1] - v1[1])
        slope2 = -(v3[0] - v1[0]) / (v3[1] - v1[1])

        delta_t_x = max(v2, v3, key=lambda x: x[0]) - min(v2, v3, key=lambda x: x[0])
        delta_t_x = delta_t_x[2] / delta_t_x[0]
        delta_t_y = v2 - v1
        delta_t_y[2] = delta_t_y[2] - delta_t_y[0] * delta_t_x
        delta_t_y = -delta_t_y[2] / delta_t_y[1]

        curX1 = v1[0]
        curX2 = v1[0]

        for i in range(int(v1[1]), int(v2[1]) - 1, -1):
            curX1_int, curX2_int = int(curX1), int(curX2)
            object_map[
                max(min(curX1_int, curX2_int), 0) : min(
                    max(curX1_int, curX2_int) + 1, self.height - 1
                ),
                i,
            ] = (
                v1[2]
                + np.arange(
                    max(min(curX1_int, curX2_int), 0) - int(v1[0]),
                    min(max(curX1_int, curX2_int), self.height - 2) - int(v1[0]) + 1,
                )
                * delta_t_x
                + (int(v1[1]) - i) * delta_t_y
            )

            curX1 += slope1
            curX2 += slope2

        return object_map

    # assume v1.y = v2.y > v3.y
    def __rasterizeTopFlatTriangle(self, v1, v2, v3):
        object_map = np.ones((self.width, self.height)) * np.inf
        slope1 = (v3[0] - v1[0]) / (v3[1] - v1[1])
        slope2 = (v3[0] - v2[0]) / (v3[1] - v2[1])

        delta_t_x = max(v1, v2, key=lambda x: x[0]) - min(v1, v2, key=lambda x: x[0])
        delta_t_x = delta_t_x[2] / delta_t_x[0]
        delta_t_y = v3 - v2
        delta_t_y[2] = delta_t_y[2] - delta_t_y[0] * delta_t_x
        delta_t_y = -delta_t_y[2] / delta_t_y[1]

        curX1 = v3[0]
        curX2 = v3[0]

        for i in range(int(v3[1]), int(v1[1]) + 1):
            curX1_int, curX2_int = int(curX1), int(curX2)
            object_map[
                max(min(curX1_int, curX2_int), 0) : min(
                    max(curX1_int, curX2_int) + 1, self.height - 1
                ),
                i,
            ] = (
                v3[2]
                + np.arange(
                    max(min(curX1_int, curX2_int), 0) - int(v3[0]),
                    min(max(curX1_int, curX2_int), self.height - 2) - int(v3[0]) + 1,
                )
                * delta_t_x
                + (int(v3[1]) - i) * delta_t_y
            )

            curX1 += slope1
            curX2 += slope2

        return object_map

    def __rasterizeTriangle(self, triangle_x_y_t):
        triangle_x_y_t = sorted(triangle_x_y_t, key=lambda x: x[1])

        v1, v2, v3 = triangle_x_y_t[0], triangle_x_y_t[1], triangle_x_y_t[2]
        # check for trivial case of bottom-flat triangle
        if v1[1] == v2[1]:
            return self.__rasterizeBottomFlatTriangle(v1, v2, v3)
        # check for trivial case of top-flat triangle
        elif v2[1] == v3[1]:
            return self.__rasterizeTopFlatTriangle(v2, v3, v1)
        # need to create artifical vertex to get a bottom-flat and top-flat triangle
        else:
            v4 = np.array(
                [(v1[0] + ((v2[1] - v1[1]) / (v3[1] - v1[1]) * (v3[0] - v1[0]))), v2[1]]
            )
            # we need to compute t for v4
            matrix_v4 = np.array([v2[:2] - v1[:2], v3[:2] - v1[:2]]).T
            vector_v4 = v4 - v1[:2]
            x_y = np.matmul(np.linalg.inv(matrix_v4), vector_v4)
            v4 = np.array(
                [
                    v4[0],
                    v4[1],
                    v1[2] + x_y[0] * (v2[2] - v1[2]) + x_y[1] * (v3[2] - v1[2]),
                ]
            )
            temp = np.minimum(
                self.__rasterizeTopFlatTriangle(v2, v4, v1),
                self.__rasterizeBottomFlatTriangle(v3, v2, v4),
            )
            return temp

    def _rasterizeTriangles(self, tri_x_y_t: np.ndarray) -> np.ndarray:
        object_map = np.empty((len(tri_x_y_t), self.width, self.height))
        for i, tri in enumerate(tri_x_y_t):
            object_map[i] = self.__rasterizeTriangle(tri)
            # make sure only vertices infront of the canvas are visible
            object_map[i] = np.where(object_map[i] < 0, np.inf, object_map[i])

        # get maximum element and add it in the end, in order to see if all values are infinity for an x, y
        max = np.where(np.isinf(object_map), -np.Inf, object_map).max()
        max_plus1 = np.ones((self.width, self.height)) * max + 1
        object_map = np.append(object_map, max_plus1[np.newaxis], axis=0)
        colors = np.array(self.triangle_fill_colors + [self.bg_color])

        # now for each pixel get the object id of the object that is closest
        pixel_id = np.argmin(object_map, axis=0)
        color_map = colors[pixel_id]

        return color_map

    def _renderProjection(
        self,
        pov: np.ndarray,
        canvas_origin: np.ndarray,
        canvas_vecX: np.ndarray,
        canvas_vecY: np.ndarray,
    ):
        tri_x_y_t = self._get2DCoordinates(pov, canvas_origin, canvas_vecX, canvas_vecY)
        return self._rasterizeTriangles(tri_x_y_t)

    def render(
        self, pov: np.ndarray, look_point: np.ndarray, algorithm="projection"
    ) -> None:
        o, v1, v2 = self._getCanvasVecs(pov, look_point)
        if algorithm == "projection":
            return self._renderProjection(pov, o, v1, v2)
        elif algorithm == "raycast":
            return self._renderRaycast(o, v1, v2)
        else:
            print(
                'Invalid algorithm for rendering, options are: "projection" and "raycast"'
            )
