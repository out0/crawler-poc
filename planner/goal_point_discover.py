#
# This class tries its best to find a good and feasible goal local WayPoint when given the
# local VehiclePose.
#
import numpy as np
from .mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from .occupancy_grid import OccupancyGrid
from .vehicle_pose import VehiclePose
from .waypoint import Waypoint
from numba import cuda
import math
from typing import List

       

class GoalPointDiscover:

    INSIDE = 0  # 0000 
    LEFT = 1    # 0001 
    RIGHT = 2   # 0010 
    BOTTOM = 4  # 0100 
    TOP = 8     # 1000 

    _og_real_size_width: int
    _og_real_size_height: int
    _minimal_distance: int
    _map_coordinate_converter: MapCoordinateConverterCarla
    _bev_center: Waypoint

    def __init__(self, og_real_size_width: int, og_real_size_height: int, minimal_distance: int) -> None:
        self._og_real_size_width = og_real_size_width
        self._og_real_size_height = og_real_size_height
        self._minimal_distance = minimal_distance
        self._map_coordinate_converter = None
        

    def initialize (self, shape: List[int]) -> None:
        self._map_coordinate_converter = MapCoordinateConverterCarla(self._og_real_size_width,  self._og_real_size_height, shape[1], shape[0])
        self._bev_center = Waypoint(int(shape[1] / 2), int(shape[0] / 2))
    
    def find_goal_waypoint(self, og: OccupancyGrid, location: VehiclePose, global_goal: VehiclePose) -> Waypoint:
        goal_on_local_coord = self._map_coordinate_converter.convert_to_waypoint(location, global_goal, clip_coordinates=False)

        shape = og.get_shape()
        position = self._find_goal_position(shape, goal_on_local_coord)

        if position == GoalPointDiscover.INSIDE:
            return goal_on_local_coord
        
        og.set_goal(goal_on_local_coord)
        result = None
        
        if position == GoalPointDiscover.TOP:
            result = self._find_best_center_for_z(og.get_frame(), 0, goal_on_local_coord, 10)

        elif position == GoalPointDiscover.BOTTOM:
            result = self._find_best_center_for_z(og.get_frame(), result.frame.shape[0] - 1, goal_on_local_coord, 10, hopping_increment=-1)

        elif position == GoalPointDiscover.LEFT:
            result = self._find_best_center_for_x(og.get_frame(), 0, goal_on_local_coord, 10)

        elif position == GoalPointDiscover.RIGHT:
            result = self._find_best_center_for_x(og.get_frame(), shape[1] - 1, goal_on_local_coord, 10, hopping_increment=-1)

        if result is None:
            min, max = self._get_search_quadrants(shape, position)
            result = self._find_goal_in_quadrant(og.get_frame(), min, max)

            if result is None:
                result = self._find_goal_in_quadrant(og.get_frame(), Waypoint(0,0), Waypoint(shape[1] - 1, shape[0] - 1))
        
        return result

    def _list_interval_center_points_for_z(self, frame: np.array, z: int) -> List[Waypoint]:
        i = 0
        start = -1
        end = -1

        res = []

        while i < frame.shape[1]:
            if frame[z, i, 2] == 0:
                if start >= 0:
                    interval = end - start
                    if interval > self._minimal_distance:
                        res.append(Waypoint(int(start + interval / 2), z))
                start = -1
                end = -1

            else:
                if start < 0:
                    start = i
                end = i
            i += 1
        return res

    def _list_interval_center_points_for_x(self, frame: np.array, x: int) -> List[Waypoint]:
        i = 0
        start = -1
        end = -1

        res = []

        while i < frame.shape[0]:
            if frame[i, x, 2] == 0:
                if start >= 0:
                    interval = end - start
                    if interval > self._minimal_distance:
                        res.append(Waypoint(x, int(start + interval / 2)))
                start = -1
                end = -1

            elif frame[i, x, 2] == 1:
                if start < 0:
                    start = i
                end = i
            i += 1
        return res

    def _compute_squared_euclidian_dist(self, p1: Waypoint, p2: Waypoint) -> float:
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return dx*dx + dz*dz

    def _find_best_center_for_z(self, frame: np.array, z: int, local_goal: Waypoint, hopping_tolerance: int, hopping_increment: int = 1) -> Waypoint:
        if hopping_tolerance == 0:
            return None
        
       
        center_points = self._list_interval_center_points_for_z(frame, z)
        c = len(center_points)

        if c == 0:
            return self._find_best_center_for_z(frame, z + hopping_increment, local_goal, hopping_tolerance - 1, hopping_increment)
        elif c == 1:
            return center_points[0]
        
        best: Waypoint = center_points[0]
        best_cost = self._compute_squared_euclidian_dist(local_goal, center_points[0])

        for i in range(1, len(center_points)):
            cost = self._compute_squared_euclidian_dist(local_goal, center_points[i])
            if cost < best_cost:
                best_cost = cost
                best = center_points[i]

        return best
    
    def _find_best_center_for_x(self, frame: np.array, x: int, local_goal: Waypoint, hopping_tolerance: int, hopping_increment: int = 1) -> Waypoint:
        if hopping_tolerance == 0:
            return None
       
        center_points = self._list_interval_center_points_for_x(frame, x)
        c = len(center_points)

        if c == 0:
            return self._find_best_center_for_x(frame, x + hopping_increment, local_goal, hopping_tolerance - 1, hopping_increment)
        elif c == 1:
            return center_points[0]
        
        best: Waypoint = center_points[0]
        best_cost = self._compute_squared_euclidian_dist(local_goal, center_points[0])

        for i in range(1, len(center_points)):
            cost = self._compute_squared_euclidian_dist(best, center_points[i])
            if cost < best_cost:
                best_cost = cost
                best = center_points[i]

        return best

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
        elif position & GoalPointDiscover.TOP:
            min = Waypoint(0,0)
            max = Waypoint(frame_shape[1] - 1, int (0.5 * frame_shape[0]))
        elif position & GoalPointDiscover.BOTTOM and position & GoalPointDiscover.LEFT:
            min = Waypoint(0,int (0.5 * frame_shape[0]))
            max = Waypoint(int (0.5 * frame_shape[1]), frame_shape[0] - 1)   
        elif position & GoalPointDiscover.BOTTOM and position & GoalPointDiscover.RIGHT:             
            min = Waypoint(int (0.5 * frame_shape[1]), int (0.5 * frame_shape[0]))
            max = Waypoint(frame_shape[1] - 1, frame_shape[0] - 1)
        elif position & GoalPointDiscover.BOTTOM:   
            min = Waypoint(0, int (0.5 * frame_shape[0]))
            max = Waypoint(frame_shape[1] - 1, frame_shape[0] - 1)         
        elif position & GoalPointDiscover.LEFT:
            min = Waypoint(0,0)
            max = Waypoint(int (0.5 * frame_shape[1]), frame_shape[0] - 1)
        elif position & GoalPointDiscover.RIGHT:
            min = Waypoint(int (0.5 * frame_shape[1]), 0)
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
            position |= GoalPointDiscover.TOP
        
        return position

    def get_map_coordinate_converter(self) -> MapCoordinateConverterCarla:
        return self._map_coordinate_converter
    