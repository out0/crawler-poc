import sys
import math
from numba import cuda
from .waypoint import Waypoint
from .vehicle_pose import VehiclePose
import numpy as np
from .mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from typing import List

DIST_FRONT = 16.3
DIST_BACK = 13.5
DIST_LEFT = 17.9
DIST_RIGHT = 17.9

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
    [55,  90,  80],
    [45,  60, 150],
    [157, 234,  50],
    [81,   0,  81],
    [150, 100, 100],
    [230, 150, 140],
    [180, 165, 180]
])


class OccupancyGrid:
    _car_width: int
    _car_length: int

    def __init__(self, car_width: int, car_length: int) -> None:
        self._car_width = car_width
        self._car_length = car_length

    @cuda.jit
    def __CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist(frame, data):
        (x, z) = cuda.grid(2)

        og_width = data[0]
        og_height = data[1]
        x_goal = data[2]
        z_goal = data[3]
        car_width = data[4]

        if x >= og_width or z >= og_height:
            return

        dz = z_goal - z
        dx = x_goal - x

        frame[z, x, 1] =  math.sqrt(dz*dz + dx*dx)
        frame[z, x, 2] = 1       

        for i in range(x - car_width/2, x + car_width/2):
            if i < 0 or i >= og_width:
                frame[z, x, 2] =  0
                return
            
            pclass = frame[z, i, 0]
            if pclass != 1 and pclass != 24 and pclass != 25 and pclass != 26 and pclass != 27 and pclass != 14:
                frame[z, x, 2] =  0
                return

    @cuda.jit
    def __CUDA__KERNEL__switch_frame_colors(image, switch_array):
        x, y = cuda.grid(2)  # 2D grid

        if x < image.shape[1] and y < image.shape[0]:
            channel_value = int(image[y, x, 0])
            for c in range(3):
                image[y, x, c] = switch_array[channel_value, c]

    def compute_distance_to_goal_feasible(self, frame: np.ndarray, goal_local_coord: Waypoint) -> None:
        d_frame = cuda.to_device(np.ascontiguousarray(frame, dtype='float'))
        threadsperblock = (16, 16)
        blockspergrid_x = (frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        d_data = cuda.to_device(np.ascontiguousarray([
            frame.shape[1],
            frame.shape[0],
            goal_local_coord.x,
            goal_local_coord.z,
            self._car_width
        ], dtype='float'))

        OccupancyGrid.__CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist[blockspergrid, threadsperblock](
            d_frame, d_data)
        
        return d_frame.copy_to_host()

    def get_color_frame(self, frame: np.ndarray) -> np.ndarray:
        cuda_colors = cuda.to_device(SEGMENTED_COLORS)
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        threadsperblock = (16, 16)
        blockspergrid_x = (frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        OccupancyGrid.__CUDA__KERNEL__switch_frame_colors[blockspergrid, threadsperblock](
            d_frame, cuda_colors)
        return d_frame.copy_to_host()

    def _compute_vector_inclination(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.atan(dy/dx)

          
    def _find_goal_between_waypoints(self, frame: np.ndarray, p1: Waypoint, p2:Waypoint) -> Waypoint:       
        best = Waypoint(0,0)
        best_dist = -1

        for z in range(p1.z, p2.z):
            for x in range(p1.x, p2.x):
                if frame[z,x,2] == 1 and (best_dist < 0 or best_dist > frame[z,x,1]):
                    best_dist = frame[z,x,1]
                    best.x = x
                    best.z = z
        if best_dist < 0:
            return None
        return best

    def _find_goal_position (self, frame_shape: List[int], goal_on_local_coord: Waypoint) -> int:
        top_left = Waypoint(0, 0)
        bottom_right =  Waypoint(frame_shape[1] - 1,  frame_shape[0] - 1)

        position = INSIDE
        if goal_on_local_coord.x < top_left.x:
            position |= LEFT
        elif goal_on_local_coord.x > bottom_right.x:
            position |= RIGHT
        if goal_on_local_coord.z > bottom_right.z:
            position |= BOTTOM
        elif goal_on_local_coord.z < top_left.z:
            position |= TOP
        
        return position

    def find_best_local_goal_waypoint(self, frame: np.ndarray, location: VehiclePose, goal: VehiclePose) -> [Waypoint, np.ndarray]:

        converter = MapCoordinateConverterCarla(DIST_LEFT + DIST_RIGHT, DIST_FRONT + DIST_BACK, frame.shape[1], frame.shape[0])
        goal_on_local_coord = converter.convert_to_waypoint(location, goal, clip_coordinates=False)
        frame = self.compute_distance_to_goal_feasible(frame, goal_on_local_coord)      
        goal_position = self._find_goal_position(frame.shape, goal_on_local_coord)

        if goal_position == INSIDE:
            return [goal_on_local_coord, frame]
        
        if goal_position & TOP:
            if goal_position & LEFT:
                min = Waypoint(0,0)
                max = Waypoint(int (0.5 * frame.shape[1]), int (0.5 * frame.shape[0]))
                local_goal = self._find_goal_between_waypoints(frame, min, max)
                if not (local_goal is None):
                    return [local_goal, frame]

            elif goal_position & RIGHT:
                min = Waypoint(int (0.5 * frame.shape[1]),0)
                max = Waypoint(frame.shape[1] - 1, int (0.5 * frame.shape[0]))
                local_goal = self._find_goal_between_waypoints(frame, min, max)
                if not (local_goal is None):
                    return [local_goal, frame]
                
                min = Waypoint(0,0)
                max = Waypoint(frame.shape[1] - 1, int (0.5 * frame.shape[0]))
                return [self._find_goal_between_waypoints(frame, min, max), frame]

        if goal_position & BOTTOM:
            if goal_position & LEFT:
                min = Waypoint(0,int (0.5 * frame.shape[0]))
                max = Waypoint(int (0.5 * frame.shape[1]), frame.shape[0] - 1)
                local_goal = self._find_goal_between_waypoints(frame, min, max)
                if not (local_goal is None):
                    return [local_goal, frame]

            elif goal_position & RIGHT:
                min = Waypoint(int (0.5 * frame.shape[1]), int (0.5 * frame.shape[0]))
                max = Waypoint(frame.shape[1] - 1, frame.shape[0] - 1)
                local_goal = self._find_goal_between_waypoints(frame, min, max)
                if not (local_goal is None):
                    return [local_goal, frame]
                
            min = Waypoint(0, int (0.5 * frame.shape[0]))
            max = Waypoint(frame.shape[1] - 1, frame.shape[0] - 1)
            return [self._find_goal_between_waypoints(frame, min, max), frame]


        if goal_position & LEFT:
            min = Waypoint(0,0)
            max = Waypoint(int (0.5 * frame.shape[1]), frame.shape[0] - 1)
            return [self._find_goal_between_waypoints(frame, min, max), frame]
            
        if goal_position & RIGHT:
            min = Waypoint(int (0.5 * frame.shape[1]), 0)
            max = Waypoint(frame.shape[1] - 1, frame.shape[0] - 1)
            return [self._find_goal_between_waypoints(frame, min, max), frame]


        return [None, frame]
    
