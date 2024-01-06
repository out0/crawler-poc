from model.waypoint import Waypoint
from model.vehicle_pose import VehiclePose
import math
import carla

class ReferencePath:
    _p1: VehiclePose
    _p2: VehiclePose

    def __init__(self, p1: VehiclePose, p2: VehiclePose) -> None:
        self._p1 = p1
        self._p2 = p2
        pass

    def get_nearest_point(self, p: VehiclePose) -> VehiclePose:
        dx= self._p2.x - self._p1.x
        dy = self._p2.y - self._p1.y
        det = dx*dx + dy*dy
        a = (dy*(p.y - self._p1.y)+dx*(p.x - self._p1.x))/det
        return VehiclePose(self._p1.x + a*dx, self._p1.y + a*dy, self._p1.heading)

    def distance_to_line(self, p: VehiclePose) -> float:
        dx = self._p2.x - self._p1.x
        dy = self._p2.y - self._p1.y
       
        num = abs(dx*(self._p1.y - p.y) - (self._p1.x - p.x)*dy)
        den = math.sqrt((dx ** 2 + dy ** 2))
        return num / den

    def compute_heading(self) -> float:
        dy = self._p2.y - self._p1.y
        dx = self._p2.x - self._p1.x
        return math.atan2(dy, dx)

class LateralController:
    __MAX_RANGE: float = 40.0
    _current_pose: callable
    _odometer: callable
    _set_steering_angle: callable
    _vehicle_length: float
    _ref_path: ReferencePath


    def __init__(self, vehicle_length: float, slam_find_current_pose: callable, odometer: callable, set_steering_angle: callable) -> None:
        self._current_pose = slam_find_current_pose
        self._odometer = odometer
        self._set_steering_angle = set_steering_angle
        self._vehicle_length = vehicle_length

    def set_reference_path(self, p1: VehiclePose, p2: VehiclePose):
        self._ref_path = ReferencePath(p1, p2)

    def __get_ref_point(self) -> VehiclePose:
        cg: VehiclePose = self._current_pose()
        a = math.radians(cg.heading)
        return VehiclePose(cg.x + math.cos(a) * self._vehicle_length, cg.y + math.sin(a) * self._vehicle_length, cg.heading)

    def __fix_range(heading: float) -> float:
        return min(
                max(heading, -LateralController.__MAX_RANGE),
                LateralController.__MAX_RANGE)


    def loop(self, dt: float, world) -> None:
        ego_ref = self.__get_ref_point()
        cg: VehiclePose = self._current_pose()

        world.debug.draw_string(carla.Location(cg.x, cg.y, 2), 'CG', draw_shadow=False,
                                       color=carla.Color(r=0, g=255, b=0), life_time=120.0,
                                       persistent_lines=True)
        
        world.debug.draw_string(carla.Location(ego_ref.x, ego_ref.y, 2), 'Ref', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
        
        nearest = self._ref_path.get_nearest_point(ego_ref)

        world.debug.draw_string(carla.Location(nearest.x, nearest.y, 2), 'N', draw_shadow=False,
                                       color=carla.Color(r=0, g=0, b=255), life_time=120.0,
                                       persistent_lines=True)

        current_speed = self._odometer()

        crosstrack_error = self._ref_path.distance_to_line(ego_ref)
        heading_error = self._ref_path.compute_heading() - math.radians(ego_ref.heading) 

        if current_speed > 0:
            new_heading = math.degrees(heading_error + math.atan(crosstrack_error / current_speed))
            new_heading = LateralController.__fix_range(new_heading)
            print(f"[{dt}] new heading computed: {new_heading}")
            self._set_steering_angle(new_heading)