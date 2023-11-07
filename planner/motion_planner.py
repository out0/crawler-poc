from planner.mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from carlasim.vehicle_hal import EgoCar
from planner.vehicle_pose import VehiclePose
from planner.waypoint import Waypoint
from planner.simple_slam import SimpleSlam
from threading import Thread
from typing import List
from planner.motion.direction_controller import DirectionController
from planner.motion.velocity_controller import VelocityController
import time

class MotionPlanner:
    _car: EgoCar
    _map_converter: MapCoordinateConverterCarla
    _async_motion_thr: Thread
    _motion_state: bool
    _next_location_list: List[VehiclePose]
    _distance_tolerance: float
    _velocity_controller: VelocityController

    def __init__(self, car: EgoCar, map_converter:  MapCoordinateConverterCarla) -> None:
        self._car = car
        self._map_converter = map_converter
        self._async_motion_thr = None
        self._motion_planner_run = False
        self._motion_state = False
        self._velocity_controller = VelocityController(car, 1.0)
        self._direction_controller = DirectionController(car, 1.0)

    def __current_pose(self) -> VehiclePose:
        l = self._car.get_location()
        return VehiclePose(l.x, l.y, self._car.get_heading())

    def __async_move(self):
        poses = self._next_location_list

        if poses is None or len(poses) == 0:
            return
        
        self._motion_state = True
        i = 0
        count = len(poses)
        self._car.forward()

        self._velocity_controller.set_speed(10)
    
        while self._motion_state and i < count:
            next_pose = poses[i]
            self._direction_controller.set_heading(next_pose.heading)
            print (f"motion to {next_pose}")
            while self.__check_is_ahead(next_pose):
                time.sleep(0.05)
            i += 1
        
        self._car.stop()
    
    def __compute_quadrant(self, pose: VehiclePose) -> int:
        if pose.heading >=0 and pose.heading <= 90:
            return 1
        if pose.heading >= -90 and pose.heading < 0:
            return 2
        if pose.heading > 90 and pose.heading <= 180:
            return 3
        if pose.heading > -180 and pose.heading < -90:
            return 4
        

    def __check_is_ahead (self, next: VehiclePose) -> bool:
        location = self.__current_pose()
        quadrant = self.__compute_quadrant(next)

        dx = next.x - location.x
        dy = next.y - location.y

        if dx == 0 and dy == 0:
            return False
        
        if quadrant == 1 and (dx >= 0 and dy >= 0):
            return True

        if quadrant == 2 and (dx >= 0 and dy <= 0):
            return True

        if quadrant == 3 and (dx <= 0 and dy >= 0):
            return True

        if quadrant == 4 and (dx <= 0 and dy <= 0):
            return True

        return False

    def cancel_motion(self) -> None:
        if self._async_motion_thr is None:
            return

        self._motion_state = False

        if self._async_motion_thr.is_alive():
            self._async_motion_thr.join(timeout=0.5)

        self._async_motion_thr = None
    
    def __compute_poses(self, location: VehiclePose, path: List[Waypoint]) -> List[VehiclePose]:
        res: List[VehiclePose] = []
        i = 1
        for p in path:
            if i % 10 == 0:
                res.append(self._map_converter.convert_to_world_pose(location, p))
            i += 1
        return res

    def move_on_path(self, location: VehiclePose, path: List[Waypoint]) -> None:
        if self._motion_state:
            self.cancel_motion()

        self._next_location_list = self.__compute_poses(location, path)
        self._current_location = location
        self._async_motion_thr = Thread(target=self.__async_move)
        self._async_motion_thr.start()

    def is_moving(self) -> bool:
        return self._motion_state

    def stop(self) -> None:
        self.cancel_motion()
        self._velocity_controller.stop()

