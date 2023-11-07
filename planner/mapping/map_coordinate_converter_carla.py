from planner.waypoint import Waypoint
from planner.vehicle_pose import VehiclePose
from .map_coordinate_converter import MapCoordinateConverter
import numpy as np
import math

class MapCoordinateConverterCarla(MapCoordinateConverter):
    _virtual_to_real_x_ratio: float
    _virtual_to_real_y_ratio: float
    _og_origin: Waypoint
    _og_width: int
    _og_height: int

    def __init__(self, og_real_size_width: float, og_real_size_height: float, og_local_size_width: int, og_local_size_height: int) -> None:
        self._virtual_to_real_x_ratio = og_real_size_width / og_local_size_width
        self._virtual_to_real_y_ratio = og_real_size_height / og_local_size_height
        self._og_origin = Waypoint(int(og_local_size_width/2), int(og_local_size_height/2))
        self._og_width = og_local_size_width
        self._og_height = og_local_size_height


    def __build_translation_mat(self, x: float, y: float) -> np.ndarray:
        return np.array([
            [1, 0 , 0],
            [0, 1, 0],
            [x, y, 1]
        ])

    def __build_rotation_mat(self, angle: float) -> np.ndarray:
        r = math.radians(angle)
        c = math.cos(r)
        s = math.sin(r)
    
        return np.array([
            [c, s, 0],
            [-s, c, 0],
            [0 , 0, 1]
        ])

    def convert_to_world_pose(self, location: VehiclePose, target : Waypoint) -> VehiclePose:

        x_for_center_on_location = target.x - self._og_origin.x
        z_for_center_on_location = self._og_origin.z - target.z 

        dx = x_for_center_on_location * self._virtual_to_real_x_ratio
        dy = z_for_center_on_location * self._virtual_to_real_y_ratio

        m = self.__build_rotation_mat(location.heading) @ self.__build_translation_mat(location.x, location.y)
        p = np.array([dy, dx, 1]) @ m

        new_heading = math.degrees(math.atan2(p[1] - location.y, p[0] - location.x))

        return VehiclePose(
            p[0] ,
            p[1] ,
            new_heading
        )
    
    def convert_to_waypoint(self, location: VehiclePose, target: VehiclePose, clip_coordinates: bool = True) -> Waypoint:
        
        m = self.__build_translation_mat(-location.x, -location.y) @ self.__build_rotation_mat(-location.heading)
        p = np.array([target.x, target.y, 1]) @ m

        # invert x and y because world cood are x vertical, y horizontal, while local coord are x horizontal, z vertical inverted
        x_for_center_on_location = p[1] / self._virtual_to_real_x_ratio
        z_for_center_on_location = -1 * p[0] / self._virtual_to_real_y_ratio

        x = math.ceil(x_for_center_on_location + self._og_origin.x)
        z = math.ceil(z_for_center_on_location + self._og_origin.z)

        if clip_coordinates:
            if x > self._og_width:
                x = self._og_width
            if z > self._og_height:
                z = self._og_height
            if x < 0:
                x = 0
            if z < 0:
                z = 0

        return Waypoint (x, z)

    