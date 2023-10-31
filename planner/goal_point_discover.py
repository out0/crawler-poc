#
# This class tries its best to find a good and feasible goal local WayPoint when given the
# local VehiclePose.
#
import numpy as np
from mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from vehicle_pose import VehiclePose
from waypoint import Waypoint
from numba import cuda
import math
from typing import List

class GoalPointDiscoverResult:
    frame: np.array
    goal: Waypoint

    def __init__(self) -> None:
        self.frame = None
        self.goal = None
        

class GoalPointDiscover:

    INSIDE = 0  # 0000 
    LEFT = 1    # 0001 
    RIGHT = 2   # 0010 
    BOTTOM = 4  # 0100 
    TOP = 8     # 1000 

    _car_width: int
    _car_length: int
    _map_coordinate_converter: MapCoordinateConverterCarla

    def __init__(self, car_width: int, car_length: int) -> None:
        self._car_width = car_width
        self._car_length = car_length
        self._map_coordinate_converter = None
        

    def initialize (self, frame: np.ndarray) -> None:
        self._map_coordinate_converter = MapCoordinateConverterCarla(36, 30, frame.shape[1], frame.shape[0])
    
    def find_goal_waypoint(self, frame: np.array, location: VehiclePose, global_goal: VehiclePose):
        result = GoalPointDiscoverResult()

        goal_on_local_coord = self._map_coordinate_converter.convert_to_waypoint(location, global_goal, clip_coordinates=False)
        result.frame = self._process_frame(frame, goal_on_local_coord) 
        position = self._find_goal_position(frame.shape, goal_on_local_coord)
        

        if position == GoalPointDiscover.INSIDE:
            result.goal = goal_on_local_coord
            return result

        min, max = self._get_search_quadrants(frame.shape, position)
        result.goal = self._find_goal_in_quadrant(frame, min, max)

    def _find_goal_in_quadrant(self, frame: np.array, min: Waypoint, max: Waypoint) -> Waypoint:
        best = Waypoint(0,0)
        best_dist = -1

        for z in range(min.z, max.z):
            for x in range(min.x, max.x):
                if frame[z,x,2] == 1 and (best_dist < 0 or best_dist > frame[z,x,1]):
                    best_dist = frame[z,x,1]
                    best.x = x
                    best.z = z
        
        if best_dist < 0:
            return None
        
        

        return best


    def _get_search_quadrants(self, frame_shape: List[int], position: int) -> [Waypoint, Waypoint]:
        if position & GoalPointDiscover.TOP and position & GoalPointDiscover.LEFT:
            min = Waypoint(0,0)
            max = Waypoint(int (0.5 * frame_shape[1]), int (0.5 * frame_shape[0]))
        elif position & GoalPointDiscover.TOP and position & GoalPointDiscover.RIGHT:
            min = Waypoint(int (0.5 * frame_shape[1]),0)
            max = Waypoint(frame_shape[1] - 1, int (0.5 * frame_shape[0]))
        elif position & GoalPointDiscover.BOTTOM and position & GoalPointDiscover.LEFT:
            min = Waypoint(0,int (0.5 * frame_shape[0]))
            max = Waypoint(int (0.5 * frame_shape[1]), frame_shape[0] - 1)    
        elif position & GoalPointDiscover.LEFT:
            min = Waypoint(0,0)
            max = Waypoint(int (0.5 * frame_shape[1]), frame_shape[0] - 1)
        elif position & GoalPointDiscover.RIGHT:
            min = Waypoint(int (0.5 * frame_shape[1]), int (0.5 * frame_shape[0]))
            max = Waypoint(frame_shape[1] - 1, frame_shape[0] - 1)
        
        return (min, max)

    def _find_goal_position (self, frame_shape: List[int], goal_on_local_coord: Waypoint) -> int:
        top_left = Waypoint(0, 0)
        bottom_right =  Waypoint(frame_shape[1] - 1,  frame_shape[0] - 1)

        position = GoalPointDiscover.INSIDE
        if goal_on_local_coord.x < top_left.x:
            position |= GoalPointDiscover.LEFT
        elif goal_on_local_coord.x > bottom_right.x:
            position |= GoalPointDiscover.RIGHT
        if goal_on_local_coord.z > bottom_right.z:
            position |= GoalPointDiscover.BOTTOM
        elif goal_on_local_coord.z < top_left.z:
            position |= GoalPointDiscover.GoalPointDiscoverTOP
        
        return position


    #### CUDA #####
    def _process_frame(self, frame: np.ndarray, goal: Waypoint) -> None:
        d_frame = cuda.to_device(np.ascontiguousarray(frame, dtype='float'))
        threadsperblock = (16, 16)
        blockspergrid_x = (frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        d_data = cuda.to_device(np.ascontiguousarray([
            frame.shape[1],
            frame.shape[0],
            goal.x,
            goal.z,
            self._car_width
        ], dtype='float'))

        GoalPointDiscover.__CUDA__KERNEL__compute_euclidian_to_goal_and_feasible_dist[blockspergrid, threadsperblock](
            d_frame, d_data)
        
        return d_frame.copy_to_host()

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