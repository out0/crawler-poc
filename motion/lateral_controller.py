from model.waypoint import Waypoint
from model.vehicle_pose import VehiclePose
from .reference_path import ReferencePath
import math

class LateralController:
    __MAX_RANGE: float = 40.0
    _current_pose: callable
    _odometer: callable
    _steering_actuator: callable
    _vehicle_length: float
    _ref_path: ReferencePath


    def __init__(self, vehicle_length: float, slam_find_current_pose: callable, odometer: callable, steering_actuator: callable) -> None:
        self._current_pose = slam_find_current_pose
        self._odometer = odometer
        self._steering_actuator = steering_actuator
        self._vehicle_length = vehicle_length
        self._ref_path = None

    def set_reference_path(self, p1: VehiclePose, p2: VehiclePose):
        self._ref_path = ReferencePath(p1, p2)

    def __get_ref_point(self) -> VehiclePose:
        cg: VehiclePose = self._current_pose()
        a = math.radians(cg.heading)
        return VehiclePose(cg.x + math.cos(a) * self._vehicle_length, cg.y + math.sin(a) * self._vehicle_length, cg.heading, 0)

    def __fix_range(heading: float) -> float:
        return min(
                max(heading, -LateralController.__MAX_RANGE),
                LateralController.__MAX_RANGE)


    def loop(self, dt: float) -> None:
        if self._ref_path is None:
            return
        
        ego_ref = self.__get_ref_point()
        current_speed = self._odometer()
        if current_speed < 0.1:
            return

        path_heading = self._ref_path.compute_heading()

        crosstrack_error = self._ref_path.distance_to_line(ego_ref)
        heading_error = path_heading - math.radians(ego_ref.heading) 

        print(f"path_heading = {math.degrees(path_heading)}, vehicle heading: {ego_ref.heading}")


        if current_speed > 0:
            new_heading = math.degrees(heading_error + math.atan(crosstrack_error / current_speed))
            new_heading = LateralController.__fix_range(new_heading)
            print(f"[lat controller] new heading: {new_heading}, ke = {crosstrack_error}, he = {heading_error}")
            self._steering_actuator(new_heading)