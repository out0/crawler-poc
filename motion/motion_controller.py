import numpy as np
from model.waypoint import Waypoint
from model.vehicle_pose import VehiclePose
from .discrete_component import DiscreteComponent
from .longitudinal_controller import LongitudinalController
from .lateral_controller import LateralController
import math
from typing import List

class MotionController (DiscreteComponent):
    _longitudinal_controller: LongitudinalController
    _lateral_controller: LateralController
    _odometer: callable
    _slam_find_current_pose: callable
    _list: List[VehiclePose]

    def __init__(self, 
                period_ms: int, 
                odometer: callable, 
                power_actuator: callable, 
                brake_actuator: callable,
                steering_actuator: callable,
                slam_find_current_pose: callable) -> None:
        super().__init__(period_ms)

        self._longitudinal_controller = LongitudinalController(
            brake_actuator=brake_actuator,
            power_actuator=power_actuator,
            odometer=odometer
        )
        self._lateral_controller = LateralController(
            vehicle_length=2,
            odometer=odometer,
            set_steering_angle=steering_actuator,
            slam_find_current_pose=slam_find_current_pose,
            
        )
        self._odometer = odometer
        self._slam_find_current_pose = slam_find_current_pose

    def set_path(self, list: List[VehiclePose]):
        self._list = list

    def _loop(self, dt: float) -> None:
        self._longitudinal_controller.loop(dt)
        self._lateral_controller.loop(dt)
        pass
