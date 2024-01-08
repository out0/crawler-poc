from model.vehicle_pose import VehiclePose
from .discrete_component import DiscreteComponent
from .longitudinal_controller import LongitudinalController
from .lateral_controller import LateralController
from .path_position_finder import PathPositionFinder
from typing import List

class MotionController (DiscreteComponent):
    _path_position_finder: PathPositionFinder
    _longitudinal_controller: LongitudinalController
    _lateral_controller: LateralController
    _on_invalid_path: callable
    _odometer: callable
    _slam_find_current_pose: callable
    _list: List[VehiclePose]
    _invalid_state: bool
    _last_pos: int

    MAX_RANGE_SQUARED = 625

    def __init__(self, 
                period_ms: int, 
                on_invalid_path: callable,
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
            steering_actuator=steering_actuator,
            slam_find_current_pose=slam_find_current_pose,
            
        )
        self._odometer = odometer
        self._slam_find_current_pose = slam_find_current_pose
        self._on_invalid_path = on_invalid_path
        self._invalid_state = False
        self._last_pos = -1

    def set_path(self, list: List[VehiclePose]):
        self._invalid_state = False
        self._list = list
        self._path_position_finder = PathPositionFinder(list)
        self._last_pos = -1

    def _loop(self, dt: float) -> None:

        if self._invalid_state:
            print ("invalid state")
            return
        
        current_pose = self._slam_find_current_pose()

        pos = self._path_position_finder.find_next_pos(self._last_pos, MotionController.MAX_RANGE_SQUARED, current_pose)

        # if pos < 0:
        #     print ("path position failed")
        #     self._on_invalid_path()
        #     self._invalid_state = True
        #     return
        
        if pos >= 0 and pos != self._last_pos:
            self._last_pos = pos
            p1 = self._list[pos]
            p2 = self._list[pos + 1]
            self._lateral_controller.set_reference_path(p1, p2)
            self._longitudinal_controller.set_speed(30)

        self._lateral_controller.loop(dt)
        self._longitudinal_controller.loop(dt)
        
    def destroy(self) -> None:
        self._longitudinal_controller.brake(1.0)
        super().destroy()