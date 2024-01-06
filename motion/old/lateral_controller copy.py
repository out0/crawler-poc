from model.waypoint import Waypoint, DistWaypoint
from model.vehicle_pose import VehiclePose
import math
import numpy as np
from .discrete_component import DiscreteComponent
from typing import List


class LateralController(DiscreteComponent):

    __KE = 3
    _current_pose: callable
    _odometer: callable
    _set_steering_angle: callable
    _waypoint_list: List[DistWaypoint]
    _last_waypoint: int
    _next_waypoint: int
    _list_size: int
    _desired_look_ahead_dist: float
    _estimated_distance_traveled: float

    def __init__(self, period_ms: int, desired_look_ahead_dist: float, slam_find_current_pose: callable, odometer: callable, set_steering_angle: callable) -> None:
        super().__init__(period_ms)
        self._current_pose = slam_find_current_pose
        self._odometer = odometer
        self._set_steering_angle = set_steering_angle
        self._waypoint_list = None
        self._prev_waypoint = 0
        self._next_waypoint = 0
        self._list_size = 0
        self._estimated_distance_traveled = 0.0
        self._desired_look_ahead_dist = desired_look_ahead_dist

    def __compute_line_terms(self, p1: Waypoint, p2: Waypoint):
        a = p1.z - p2.z
        b = p2.x - p1.x
        c = (p1.x - p2.x) * p1.z + (p2.z - p1.z) * p1.x
        return a, b, c

    def __compute_sterring_angle(self, current: VehiclePose, previous: DistWaypoint, next: DistWaypoint, velocity: float):
        a, b, c = self.__compute_line_terms(previous, next)
        crosstrack_error = (a * current.x + b * current.y +
                            c) / math.sqrt(a ** 2 + b ** 2)
        sterring_error = math.atan(
            LateralController.__KE * crosstrack_error / velocity)
        if a == 0 and b == 0:
            heading_error = 0
        elif b == 0:
            heading_error = -1 * math.pi
        else:
            heading_error = math.atan(-a / b) - math.radians(current.heading)
        return heading_error + sterring_error

    def set_waypoint_list(self, list: List[DistWaypoint]):
        self._waypoint_list = list
        self._list_size = len(list)
        self._prev_waypoint = 0
        self._next_waypoint = 0        
        self._current_estimated_distance = 0.0

    # TEMP, should not be used!! VehiclePose != Waypoint
    def __compute_distance(self, p1: Waypoint, p2:Waypoint):
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.sqrt(dx ** 2 + dz ** 2)
        

    def __find_waypoint_reference_pair(self, velocity: float, dt: float, pose: VehiclePose) -> (int, int):
        self._estimated_distance_traveled += velocity * dt
        dist_to_correct_course = self._estimated_distance_traveled + self._desired_look_ahead_dist

        prev = self._prev_waypoint
        while prev < self._list_size:
             # temp, remove!!
            if self._waypoint_list[prev].dist < 0:
               
                self._waypoint_list[prev].dist = self.__compute_distance(Waypoint(pose.x, pose.y), self._waypoint_list[prev])
            if self._estimated_distance_traveled - self._waypoint_list[prev].dist <= dist_to_correct_course:
                break
            prev += 1
        
        next = self._next_waypoint
        while next < self._list_size:
             # temp, remove!!
            if self._waypoint_list[next].dist < 0:
                self._waypoint_list[next].dist = self.__compute_distance(Waypoint(pose.x, pose.y), self._waypoint_list[next])
            if self._waypoint_list[next].dist - self._estimated_distance_traveled >= dist_to_correct_course:
                break
            next += 1
        
        if prev >= self._list_size or next >= self._list_size:
            return -1, -1
        
        return prev, next

    def _loop(self, dt: float) -> None:
        if self._waypoint_list is None:
            return
        
        pose: VehiclePose = self._current_pose()
        velocity: float = self._odometer()

        prev, next = self.__find_waypoint_reference_pair(velocity, dt, pose)
        if prev == -1 or next == -1:
            # desperately in need for replanning
            return
        
       

        angle = self.__compute_sterring_angle(pose, self._waypoint_list[prev], self._waypoint_list[next], velocity)


        angle = math.degrees(angle)

        print (f"ref points {prev} -> {next} | computed angle: {angle}")
        
        if angle > 40:
            angle = 40
        elif angle < -40:
            angle = -40
        
        self._set_steering_angle(angle)