import sys
import math
from numba import cuda
from planner.waypoint import Waypoint
import numpy as np
import cv2

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
        (x, y ) = cuda.grid(2)
        
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
        frame[y , x, 2] = 0

        if x < car_width or x > bev_width - car_width:
            return

        i = int(x - car_width)
        feasible = 1

        c = 0
        while feasible == 1 and i < bev_width and c < car_width:
            pclass = frame[y,i,0]         
            feasible = pclass == 1 or pclass == 24 or pclass == 25 or pclass == 26 or pclass == 27
            i += 1
            c += 1

        if c < car_width:
            feasible = 0

        frame[y,x,2] = feasible

    @cuda.jit
    def __CUDA__KERNEL__switch_frame_colors(image, switch_array):
        x, y = cuda.grid(2)  # 2D grid

        if x < image.shape[1] and y < image.shape[0]:
            channel_value = int(image[y, x, 0])
            if 0 <= channel_value < 20:
                for c in range(3):
                    image[y, x, c] = switch_array[channel_value, c]
    


    def compute_distance_to_goal_feasible(self, goal: Waypoint) -> None:
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
            goal.z
        ], dtype='float'))

        OccupancyGrid.__CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist[blockspergrid, threadsperblock](d_frame, d_data)
        self._frame = d_frame.copy_to_host()
    
    def get_color_frame(self) -> np.ndarray:
        cuda_colors = cuda.to_device(SEGMENTED_COLORS)
        d_frame = cuda.to_device(np.ascontiguousarray(self._frame))
        threadsperblock = (16, 16)
        blockspergrid_x = (self._frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (self._frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        OccupancyGrid.__CUDA__KERNEL__switch_frame_colors[blockspergrid, threadsperblock](d_frame, cuda_colors)
        return d_frame.copy_to_host()
    
        
