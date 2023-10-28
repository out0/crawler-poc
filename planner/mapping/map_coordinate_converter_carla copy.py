from planner.waypoint import Waypoint
from planner.vehicle_pose import VehiclePose
from .map_coordinate_converter import MapCoordinateConverter
import numpy as np
import math

class MapCoordinateConverterCarla(MapCoordinateConverter):
    _virtual_to_real_x_ratio: float
    _virtual_to_real_y_ratio: float
    _bev_origin: Waypoint

    def __init__(self, og_world_width: float, og_world_height: float, og_width: int, og_height: int) -> None:
        self._virtual_to_real_x_ratio = og_world_width / og_width
        self._virtual_to_real_y_ratio = og_world_height / og_height
        self._bev_origin = Waypoint(int(og_width/2), int(og_height/2))


    def build_coordinate_transf_mat(self, x: float, y: float, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
    
        return np.array([
            [1, 0 , 0],
            [0, 1, 0],
            [x, y, 1]
        ]) @ np.array([
            [c, s, 0],
            [-s, c, 0],
            [0 , 0, 1]
        ])

    def convert_to_world_pose(self, location: VehiclePose, target : Waypoint) -> VehiclePose:

        x_for_center_on_location = target.x - self._bev_origin.x
        z_for_center_on_location = self._bev_origin.z - target.z 

        dx = x_for_center_on_location * self._virtual_to_real_x_ratio
        dy = z_for_center_on_location * self._virtual_to_real_y_ratio

        m = self.build_coordinate_transf_mat(location.x, location.y, location.heading)

        p = np.array([
            dx, dy, 1
        ]) @ m

        new_heading = math.atan2(p[1] - location.y, p[0] - location.x)

        return VehiclePose(
            p[0],
            p[1],
            new_heading
        )
    
    def convert_to_waypoint(self, location: VehiclePose, target: VehiclePose) -> Waypoint:

        p = np.array([
            target.x, target.y, 1
        ]) @ self.build_coordinate_transf_mat(-location.x, -location.y, -location.heading)

        x_for_center_on_location = p[0] / self._virtual_to_real_x_ratio
        z_for_center_on_location = p[1] / self._virtual_to_real_y_ratio

        x = x_for_center_on_location + self._bev_origin.x
        z = self._bev_origin.z - z_for_center_on_location

        return Waypoint (x, z)

    
    
