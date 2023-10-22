import sys
import math
from numba import cuda
from .waypoint import Waypoint
from .vehicle_pose import VehiclePose
import numpy as np
from .mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla

DIST_FRONT = 16.3
DIST_BACK = 13.5
DIST_LEFT = 17.9
DIST_RIGHT = 17.9

SEGMENTED_COLORS = np.array([
    [0,   0,   0],
    [128,  64, 128],
    [244,  35, 232],
    [70,  70,  70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170,  30],
    [220, 220,   0],
    [107, 142,  35],
    [152, 251, 152],
    [70, 130, 180],
    [220,  20,  60],
    [255,   0,   0],
    [0,   0, 142],
    [0,   0,  70],
    [0,  60, 100],
    [0,  80, 100],
    [0,   0, 230],
    [119,  11,  32],
    [110, 190, 160],
    [170, 120,  50],
    [55,  90,  80],
    [45,  60, 150],
    [157, 234,  50],
    [81,   0,  81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180]
])


class OccupancyGrid:
    _frame: np.ndarray
    _car_width: int
    _car_length: int

    def __init__(self, frame: np.ndarray,  car_width: int, car_length: int) -> None:
        self._frame = frame
        self._car_width = car_width
        self._car_length = car_length

    @cuda.jit
    def __CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist(frame, data):
        (x, y) = cuda.grid(2)

        bev_width = data[0]
        bev_height = data[1]
        real_bev_size_width = data[2]
        real_bev_size_height = data[3]
        car_width = data[4]
        car_length = data[5]
        x_goal = data[6]
        y_goal = data[7]

        if x >= bev_width or y >= bev_height:
            return

        x_ratio = (real_bev_size_width - car_width)/bev_width
        y_ratio = (real_bev_size_height - car_length)/bev_height

        xg = x_ratio * x - x_goal
        yg = y_ratio * y - y_goal

        frame[y, x, 1] = math.sqrt(xg*xg + yg*yg)
        frame[y, x, 2] = 0

        if x < car_width or x > bev_width - car_width:
            return

        i = int(x - car_width)
        feasible = 1

        c = 0
        while feasible == 1 and i < bev_width and c < car_width:
            pclass = frame[y, i, 0]
            feasible = pclass == 1 or pclass == 24 or pclass == 25 or pclass == 26 or pclass == 27
            i += 1
            c += 1

        if c < car_width:
            feasible = 0

        frame[y, x, 2] = feasible

    @cuda.jit
    def __CUDA__KERNEL__switch_frame_colors(image, switch_array):
        x, y = cuda.grid(2)  # 2D grid

        if x < image.shape[1] and y < image.shape[0]:
            channel_value = int(image[y, x, 0])
            for c in range(3):
                image[y, x, c] = switch_array[channel_value, c]

    # TODO: find a way of cope with vehicle's movement changing the frame

    @cuda.jit
    def __CUDA__KERNEL__check_path_feasible(frame, path, data):
        x, y = cuda.grid(2)  # 2D grid

        if x >= frame.shape[1] or y >= frame.shape[0]:
            return

        car_width = data[0]

        i = 0
        while i < path.shape[0]:
            point_x = path[i][0]
            point_y = path[i][1]

            if x == point_x and y == point_y:
                c = 0
                feasible = 1
                while feasible == 1 and i < frame.shape[0] and c < car_width:
                    pclass = frame[y, i, 0]
                    feasible = pclass == 1 or pclass == 24 or pclass == 25 or pclass == 26 or pclass == 27
                    i += 1
                    c += 1

                if c < car_width:
                    feasible = 0

                if feasible == 0:
                    data[1] = 0

            i += 1

    def compute_distance_to_goal_feasible(self, goal: VehiclePose) -> None:
        d_frame = cuda.to_device(np.ascontiguousarray(self._frame))
        threadsperblock = (16, 16)
        blockspergrid_x = (self._frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (self._frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        d_data = cuda.to_device(np.ascontiguousarray([
            self._frame.shape[0],
            self._frame.shape[1],
            DIST_LEFT + DIST_RIGHT,
            DIST_FRONT + DIST_BACK,
            self._car_width,
            self._car_length,
            goal.x,
            goal.y
        ], dtype='float'))

        OccupancyGrid.__CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist[blockspergrid, threadsperblock](
            d_frame, d_data)
        self._frame = d_frame.copy_to_host()

    def get_color_frame(self) -> np.ndarray:
        cuda_colors = cuda.to_device(SEGMENTED_COLORS)
        d_frame = cuda.to_device(np.ascontiguousarray(self._frame))
        threadsperblock = (16, 16)
        blockspergrid_x = (self._frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (self._frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        OccupancyGrid.__CUDA__KERNEL__switch_frame_colors[blockspergrid, threadsperblock](
            d_frame, cuda_colors)
        return d_frame.copy_to_host()

    def _compute_vector_inclination(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.atan2(dy, dx)

    
    def _find_best_local_goal_waypoint_top(self, location: VehiclePose, goal: VehiclePose) -> Waypoint:
        pass
    def _find_best_local_goal_waypoint_left(self, location: VehiclePose, goal: VehiclePose) -> Waypoint:
        pass
    def _find_best_local_goal_waypoint_right(self, location: VehiclePose, goal: VehiclePose) -> Waypoint:
        pass
    def _find_best_local_goal_waypoint_bottom(self, location: VehiclePose, goal: VehiclePose) -> Waypoint:
        pass                


    def find_best_local_goal_waypoint(self, location: VehiclePose, goal: VehiclePose) -> Waypoint:
        self.compute_distance_to_goal_feasible(goal)

        p = self._compute_vector_inclination(location,  goal)

        converter = MapCoordinateConverterCarla(DIST_LEFT + DIST_RIGHT, DIST_FRONT + DIST_BACK, self._frame.shape[1], self._frame.shape[0])

        top_left = converter.convert_to_world_pose(location, Waypoint(0, 0))
        top_right = converter.convert_to_world_pose(location, Waypoint(self._frame.shape[1] - 1, 0))
        bottom_left = converter.convert_to_world_pose(location, Waypoint(0, self._frame.shape[0] - 1))
        bottom_right = converter.convert_to_world_pose(location, Waypoint(self._frame.shape[1] - 1,  self._frame.shape[0] - 1))

        deriv_top_left = self._compute_vector_inclination(location, top_left);
        deriv_top_right = self._compute_vector_inclination(location, top_right);
        deriv_bottom_left = self._compute_vector_inclination(location, bottom_left);
        deriv_bottom_right = self._compute_vector_inclination(location, bottom_right);

        # inverse = False
        # if p < 0:
        #     inverse = True
        #     p = abs(p)


        print (f"goal: {goal}")
        print (f"top_left: {top_left}")
        print (f"top_right: {top_right}")
        print (f"bottom_left: {bottom_left}")
        print (f"bottom_right: {bottom_right}")



        print (f"derivada v-base: {p}")
        print (f"derivada top_left: {deriv_top_left}")
        print (f"derivada top_right: {deriv_top_right}")
        print (f"derivada bottom_left: {deriv_bottom_left}")
        print (f"derivada bottom_right: {deriv_bottom_right}")

        if p >= deriv_top_left and p <= deriv_top_right:
            if inverse:
                print (f"point {goal} is on BOTTOM: between {bottom_left} and {bottom_right}")
                return self._find_best_local_goal_waypoint_bottom(location, goal)
            
            print (f"point {goal} is on TOP: between {top_left} and {top_right}")
            return self._find_best_local_goal_waypoint_top(location, goal)
        
        if p >= deriv_bottom_left and p <= deriv_top_left:
            print (f"point {goal} is on LEFT: between {bottom_left} and {top_left}")
            return self._find_best_local_goal_waypoint_left(location, goal)

        if p >= deriv_bottom_right and p <= deriv_top_right:
            print (f"point {goal} is on RIGHT: between {bottom_right} and {top_right}")
            return self._find_best_local_goal_waypoint_right(location, goal)

        if p >= deriv_bottom_left and p <= deriv_bottom_right:
            if inverse:
                print (f"point {goal} is on TOP: between {top_left} and {top_right}")
                return self._find_best_local_goal_waypoint_top(location, goal)

            print (f"point {goal} is on BOTTOM: between {bottom_left} and {bottom_right}")
            return self._find_best_local_goal_waypoint_bottom(location, goal)
        
        print ("no classification found")

        return None