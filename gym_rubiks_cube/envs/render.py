import numpy as np
from typing import List, Tuple
from numba import njit, prange

from gym_rubiks_cube.envs.functions import RotationMatrix3D, getQuadrant
import gym_rubiks_cube.envs.objects3D as o3


@njit
def rasterizeBottomFlatTriangle(v1, v2, v3, width, height):
    """assumes v1.y < v2.y = v3.y"""
    object_map = np.ones((width, height), dtype=np.float32) * np.inf
    v1[:2], v2[:2], v3[:2] = np.floor(v1[:2]), np.floor(v2[:2]), np.floor(v3[:2])
    if v1[1] == v2[1] or v1[1] == v2[1] or v2[0] == v3[0]:
        return object_map
    slope1 = -(v2[0] - v1[0]) / np.floor(v2[1] - v1[1])
    slope2 = -(v3[0] - v1[0]) / np.floor(v3[1] - v1[1])

    if v2[0] > v3[0]:
        delta_t_x = v2 - v3
    else:
        delta_t_x = v3 - v2
    delta_t_x = delta_t_x[2] / delta_t_x[0]
    delta_t_y = v2 - v1
    delta_t_y[2] = delta_t_y[2] - delta_t_y[0] * delta_t_x
    delta_t_y = -delta_t_y[2] / delta_t_y[1]

    curX1 = v1[0]
    curX2 = v1[0]

    for i in range(int(v1[1]), int(v2[1]) - 1, -1):
        curX1_int, curX2_int = int(curX1), int(curX2)
        temp = (
            v1[2]
            + np.arange(
                max(min(curX1_int, curX2_int), 0) - int(v1[0]),
                min(max(curX1_int, curX2_int), height - 2) - int(v1[0]) + 1,
            )
            * delta_t_x
            + (int(v1[1]) - i) * delta_t_y
        )
        object_map[
            max(min(curX1_int, curX2_int), 0) : min(
                max(curX1_int, curX2_int) + 1, height - 1
            ),
            i,
        ] = np.where(
            temp < 0, np.inf, temp
        )  # make sure only vertices infront of the canvas are visible

        curX1 += slope1
        curX2 += slope2

    return object_map


@njit
def rasterizeTopFlatTriangle(v1, v2, v3, width, height):
    """assumes v1.y = v2.y < v3.y"""
    object_map = np.ones((width, height), dtype=np.float32) * np.inf
    v1[:2], v2[:2], v3[:2] = np.floor(v1[:2]), np.floor(v2[:2]), np.floor(v3[:2])
    if v3[1] == v1[1] or v3[1] == v2[1] or v1[0] == v2[0]:
        return object_map
    slope1 = (v3[0] - v1[0]) / np.floor(v3[1] - v1[1])
    slope2 = (v3[0] - v2[0]) / np.floor(v3[1] - v2[1])

    minX, maxX = min(v1[0], v2[0]), max(v1[0], v2[0])

    if v1[0] > v2[0]:
        delta_t_x = v1 - v2
    else:
        delta_t_x = v2 - v1
    delta_t_x = delta_t_x[2] / delta_t_x[0]
    delta_t_y = v3 - v2
    delta_t_y[2] = delta_t_y[2] - delta_t_y[0] * delta_t_x
    delta_t_y = -delta_t_y[2] / delta_t_y[1]

    curX1 = v3[0]
    curX2 = v3[0]

    for i in range(int(v3[1]), int(v1[1]) + 1):
        curX1_int, curX2_int = int(curX1), int(curX2)
        temp = (
            v3[2]
            + np.arange(
                max(min(curX1_int, curX2_int), 0) - int(v3[0]),
                min(max(curX1_int, curX2_int), height - 2) - int(v3[0]) + 1,
            )
            * delta_t_x
            + (int(v3[1]) - i) * delta_t_y
        )
        object_map[
            max(min(curX1_int, curX2_int), 0) : min(
                max(curX1_int, curX2_int), height - 2
            )
            + 1,
            i,
        ] = np.where(
            temp < 0, np.inf, temp
        )  # make sure only vertices infront of the canvas are visible

        curX1 += slope1
        curX2 += slope2

    return object_map


@njit
def rasterizeTriangle(triangle_x_y_t, width, height):
    triangle_x_y_t = sorted(triangle_x_y_t, key=lambda x: x[1])

    v1, v2, v3 = triangle_x_y_t[0], triangle_x_y_t[1], triangle_x_y_t[2]
    if int(v1[1]) == int(v2[1]) and int(v2[1]) == int(v3[1]):
        return np.ones((width, height)) * np.inf
    # check for trivial case of bottom-flat triangle
    elif v1[1] == v2[1]:
        return rasterizeBottomFlatTriangle(v3, v1, v2, width, height)
    # check for trivial case of top-flat triangle
    elif v2[1] == v3[1]:
        return rasterizeTopFlatTriangle(v2, v3, v1, width, height)
    # need to create artifical vertex to get a bottom-flat and top-flat triangle
    else:
        v4 = np.array(
            [(v1[0] + ((v2[1] - v1[1]) / (v3[1] - v1[1]) * (v3[0] - v1[0]))), v2[1]]
        )
        # we need to compute t for v4
        matrix_v4 = np.array(
            [[v2[0] - v1[0], v3[0] - v1[0]], [v2[1] - v1[1], v3[1] - v1[1]]]
        )
        vector_v4 = v4 - v1[:2]
        x_y = np.dot(np.linalg.inv(matrix_v4), vector_v4)
        v4 = np.array(
            [
                v4[0],
                v4[1],
                v1[2] + x_y[0] * (v2[2] - v1[2]) + x_y[1] * (v3[2] - v1[2]),
            ]
        )
        return np.minimum(
            rasterizeTopFlatTriangle(v2, v4, v1, width, height),
            rasterizeBottomFlatTriangle(v3, v2, v4, width, height),
        )


@njit(parallel=True)
def rasterizeTrianglesHelp(tri_x_y_t, width, height):
    object_map = np.empty((len(tri_x_y_t) + 1, width, height), dtype=np.float32)
    for i in prange(len(tri_x_y_t)):
        object_map[i] = rasterizeTriangle(tri_x_y_t[i], width, height)

    return object_map


class Scene:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        objects_to_render: List[o3.Renderable],
        canvas_distance: float,
        bg_color,
    ) -> None:
        # save general parameters
        self.width = screen_width
        self.height = screen_height
        self.canvas_distance = canvas_distance
        self.bg_color = np.array(bg_color)

        # save render parameters
        self.switch = True
        self.last_quadrant = None

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

    # something wrong with this, not sure what (don't really wanna spend time fixing it though cause it's too slow anyway)
    def _renderRaycast(
        self,
        pov: np.ndarray,
        canvas_origin: np.ndarray,
        canvas_vecX: np.ndarray,
        canvas_vecY: np.ndarray,
    ):
        # initialize color map
        rgb_map = np.empty((self.width, self.height, 3), dtype=np.int16)
        t = np.arange(self.height) - self.height / 2
        canvas_vec2 = np.repeat(canvas_vecY[np.newaxis, :], self.height, axis=0)
        objects_vec1_repeated = np.repeat(
            self.triangle_vec1s[np.newaxis, :, :, np.newaxis], self.height, axis=0
        )
        objects_vec2_repeated = np.repeat(
            self.triangle_vec2s[np.newaxis, :, :, np.newaxis], self.height, axis=0
        )
        vector_b_repeated = np.repeat(
            (pov - self.triangle_origins)[np.newaxis, :, :, np.newaxis],
            self.height,
            axis=0,
        )

        # append background color to object colors at the end for later
        objects_color = np.append(
            self.triangle_fill_colors, self.bg_color[np.newaxis, :], axis=0
        )

        for i in range(self.width):
            # create rays
            # create ray direction (point on canvas - pov)
            ray_direction = (
                canvas_origin
                + canvas_vecX * (i - self.width / 2)
                + canvas_vec2 * t[:, np.newaxis]
                - pov
            )
            # now get intersection of ray with objects (rectangles)
            # create matrix that needs to be inverted with shape (height, N, 3, 3), with N = number of squares in world
            ray_direction_repeated = np.repeat(
                ray_direction[:, np.newaxis, :, np.newaxis],
                len(self.triangle_origins),
                axis=1,
            )
            matrices_a = np.append(
                objects_vec1_repeated,
                np.append(objects_vec2_repeated, -ray_direction_repeated, axis=3),
                axis=3,
            )

            inverse = np.linalg.pinv(matrices_a)
            gamma_phi_t = np.matmul(inverse, vector_b_repeated)

            outside_of_rect = (
                (gamma_phi_t[:, :, 0, 0] < 0)
                | (gamma_phi_t[:, :, 0, 0] > 1)
                | (gamma_phi_t[:, :, 1, 0] < 0)
                | (gamma_phi_t[:, :, 1, 0] + gamma_phi_t[:, :, 0, 0] > 1)
                | (gamma_phi_t[:, :, 2, 0] <= 0)
            )
            valid_t = np.where(outside_of_rect, np.inf, gamma_phi_t[:, :, 2, 0])
            obj_seen = np.argmin(valid_t, axis=1)
            all_inf = (valid_t == np.inf).all(axis=1)
            color = np.where(all_inf, -1, obj_seen)

            rgb_map[i] = objects_color[color]

        return rgb_map

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
        tri_x_y_t = np.matmul(np.linalg.inv(matrix_tri), vector_tri)[:, :, :, 0]

        return tri_x_y_t

    def _rasterizeTriangles(self, tri_x_y_t: np.ndarray) -> np.ndarray:
        object_map = rasterizeTrianglesHelp(tri_x_y_t, self.width, self.height)

        # append (non infinity-element) in the end, in order to see if all values would be infinity, then we render the background color
        # assume that this is the biggest element (distance of an object would need to be greater than it (which shouldn't ever happen))
        max_plus1 = np.ones((self.width, self.height)) * 100000
        object_map[-1] = max_plus1

        colors = np.array(self.triangle_fill_colors + [self.bg_color], dtype=np.uint8)

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
            return self._renderRaycast(pov, o, v1, v2)
        else:
            print(
                'Invalid algorithm for rendering, options are: "projection" and "raycast"'
            )

    def rotateObjects(
        self, objects_ids: List[int], axis: int, rotation_angle: float
    ) -> None:
        """rotates objects given by their id around the axis (0, 1 or 2) with the given rotation angle (around the coordinate center)"""
        rot = RotationMatrix3D()
        for obj_id in objects_ids:
            triangles = self.objects[obj_id]
            for elem in triangles:
                self.triangle_origins[elem] = rot(
                    self.triangle_origins[elem], axis, rotation_angle
                )
                self.triangle_vec1s[elem] = rot(
                    self.triangle_vec1s[elem], axis, rotation_angle
                )
                self.triangle_vec2s[elem] = rot(
                    self.triangle_vec2s[elem], axis, rotation_angle
                )

    def _getObjectsCenters(self):
        """for debugging"""
        points = [[] for i in range(len(self.objects))]
        for key, value in self.objects.items():
            for ids in value:
                points[key].append(self.triangle_origins[ids])
                points[key].append(
                    self.triangle_origins[ids] + self.triangle_vec1s[ids]
                )
                points[key].append(
                    self.triangle_origins[ids] + self.triangle_vec2s[ids]
                )
            points[key] = np.mean(points[key], axis=0)

        out = np.empty((3, 3, 3))
        arrZY = [100, 0, -100]
        arrZX = [-100, 0, 100]
        for i, elem in enumerate(points):
            index1, index2, index3 = (
                arrZY.index(np.round(elem[2])),
                arrZY.index(np.round(elem[1])),
                arrZX.index(np.round(elem[0])),
            )
            out[index1, index2, index3] = i

        return out
