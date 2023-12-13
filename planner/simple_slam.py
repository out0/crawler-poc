import math
from .vehicle_pose import VehiclePose
from .waypoint import Waypoint
from carlasim.vehicle_hal import EgoCar

class SimpleSlam:
    _x_ratio: float
    _y_ratio: float
    _x_center: float
    _z_center: float
    _bev_width: int
    _bev_height: int

    def __init__(self, bev_width: int, bev_height, car_width: float, car_length: float) -> None:
        self._x_ratio = (16 - car_width) / bev_width
        self._y_ratio = (16 - car_length) / bev_height
        self._bev_width = bev_width
        self._bev_height = bev_height
        self._x_center = bev_width/2
        self._z_center = bev_height/2

    def _compute_vect_size(self, x: float, z: float) -> float:
        dx = x - self._x_center
        dz = z - self._z_center
        return math.sqrt(dx*dx + dz*dz)
    
    def _compute_bev_angle(self, x: float, z: float) -> float:
        dx = abs(x - self._x_center)
        dz = abs(z - self._z_center)
        if dz == 0:
            return 0
        return (math.atan(dx/dz) * 180) / math.pi
        
    def _compute_next_pose(self, pose: VehiclePose, bev_waypoint: Waypoint) -> VehiclePose:       
        vect_size = self._compute_vect_size(bev_waypoint.x, bev_waypoint.z)

        a = pose.heading
        if bev_waypoint.x < self._x_center:
            a -= self._compute_bev_angle(bev_waypoint.x, bev_waypoint.z)
        else:
            a += self._compute_bev_angle(bev_waypoint.x, bev_waypoint.z)

        ra = (math.pi * a) / 180

        xp = vect_size * math.cos(ra) * self._x_ratio + pose.x
        yp = vect_size * math.sin(ra) * self._y_ratio + pose.y
        return (xp, yp, a)

    def estimate_next_pose(self, car: EgoCar, bev_waypoint: Waypoint) -> VehiclePose:       
        vect_size = self._compute_vect_size(bev_waypoint.x, bev_waypoint.z)

        self_pose = self.get_current_pose(car)

        a = self_pose.heading
        if bev_waypoint.x < self._x_center:
            a -= self._compute_bev_angle(bev_waypoint.x, bev_waypoint.z)
        else:
            a += self._compute_bev_angle(bev_waypoint.x, bev_waypoint.z)

        ra = (math.pi * a) / 180

        xp = vect_size * math.cos(ra) * self._x_ratio + self_pose.x
        yp = vect_size * math.sin(ra) * self._y_ratio + self_pose.y
        return (xp, yp, a)


    def get_current_pose(self, car: EgoCar) -> VehiclePose:
        carla_location = car.get_location()
        a = car.get_yaw()
        return VehiclePose(carla_location.x, carla_location.y, a)
