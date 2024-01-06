import sys
import math
from numba import cuda
from model.waypoint import Waypoint
import numpy as np

INSIDE = 0  # 0000 
LEFT = 1    # 0001 
RIGHT = 2   # 0010 
BOTTOM = 4  # 0100 
TOP = 8     # 1000 

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
    [55,  90,  80],     # other
    [45,  60, 150],
    [157, 234,  50],
    [81,   0,  81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180]
])



class OccupancyGrid:
    _minimal_distance: int
    _frame: np.array
    _goal_point: Waypoint
    _cuda_frame: any


    def __init__(self, frame: np.array, minimal_distance: int) -> None:
        self._minimal_distance = minimal_distance
        self._frame = frame
        self._goal_point = None
        self._cuda_frame = None


    def set_goal(self, goal: Waypoint) -> None:
        self._goal_point = goal

        if self._cuda_frame is None:
            self._cuda_frame = cuda.to_device(np.ascontiguousarray(self._frame, dtype='float'))

        OccupancyGrid.__CUDA_compute_euclidian_distance_and_feasible_dist(self._cuda_frame, goal, self._minimal_distance)

        self._frame = self._cuda_frame.copy_to_host()
        


    @cuda.jit
    def __CUDA__KERNEL__switch_frame_colors(image, switch_array):
        x, y = cuda.grid(2)  # 2D grid

        if x < image.shape[1] and y < image.shape[0]:
            channel_value = int(image[y, x, 0])
            for c in range(3):
                image[y, x, c] = switch_array[channel_value, c]

    def __CUDA_compute_euclidian_distance_and_feasible_dist(d_frame: np.ndarray, goal: Waypoint, minimal_distance: int) -> np.ndarray:
        
        threadsperblock = (16, 16)
        blockspergrid_x = (d_frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (d_frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        d_data = cuda.to_device(np.ascontiguousarray([
            d_frame.shape[1],
            d_frame.shape[0],
            goal.x,
            goal.z,
            minimal_distance
        ], dtype='float'))

        OccupancyGrid.__CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist[blockspergrid, threadsperblock](
            d_frame, d_data)

    @cuda.jit
    def __CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist(frame, data):
        (x, z) = cuda.grid(2)

        og_width = data[0]
        og_height = data[1]
        x_goal = data[2]
        z_goal = data[3]
        minimal_distance = data[4]

        if x >= og_width or z >= og_height:
            return

        dz = z_goal - z
        dx = x_goal - x

        frame[z, x, 1] = math.sqrt(dz*dz + dx*dx)
        frame[z, x, 2] = 1

        lower_range_x = x - minimal_distance
        upper_range_x = x + minimal_distance
        lower_range_z = z - minimal_distance
        upper_range_z = z + minimal_distance
      

        if lower_range_x < 0:
            lower_range_x = 0
            pclass = frame[z, lower_range_x, 0]
            if not(pclass == 1\
                or pclass == 6\
                or pclass == 14\
                or pclass == 22\
                or pclass == 24\
                or pclass == 25\
                or pclass == 26\
                or pclass == 27):
                frame[z, x, 2] = 0
                return
        
        if lower_range_z < 0:
            lower_range_z = 0
            pclass = frame[lower_range_z, x, 0]
            if not(pclass == 1\
                or pclass == 6\
                or pclass == 14\
                or pclass == 22\
                or pclass == 24\
                or pclass == 25\
                or pclass == 26\
                or pclass == 27):
                frame[z, x, 2] = 0
                return


        if upper_range_x >= og_width:
            upper_range_x = int(og_width - 1)
            pclass = frame[z, upper_range_x, 0]
            if not(pclass == 1\
                or pclass == 6\
                or pclass == 14\
                or pclass == 22\
                or pclass == 24\
                or pclass == 25\
                or pclass == 26\
                or pclass == 27):
                frame[z, x, 2] = 0
                return     
        
        if upper_range_z >= og_height:
            upper_range_z = int(og_height - 1)
            pclass = frame[upper_range_z, x, 0]
            if not(pclass == 1\
                or pclass == 6\
                or pclass == 14\
                or pclass == 22\
                or pclass == 24\
                or pclass == 25\
                or pclass == 26\
                or pclass == 27):
                frame[z, x, 2] = 0
                return              

        for i in range(lower_range_z, upper_range_z + 1):
            for j in range(lower_range_x, upper_range_x + 1):
                pclass = frame[i, j, 0]
                if not(pclass == 1\
                    or pclass == 6\
                    or pclass == 14\
                    or pclass == 22\
                    or pclass == 24\
                    or pclass == 25\
                    or pclass == 26\
                    or pclass == 27):
                    frame[z, x, 2] = 0
                    return                    
   
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

    
    def get_frame(self) -> np.array:
        return self._frame
        
    def get_shape(self):
        if self._frame is None:
            return None
        return self._frame.shape