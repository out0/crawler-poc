from planner.waypoint import Waypoint
from planner.vehicle_pose import VehiclePose
from .map_coordinate_converter import MapCoordinateConverter

class MapCoordinateConverterCarla(MapCoordinateConverter):
    _virtual_to_real_x_ratio: float
    _virtual_to_real_y_ratio: float
    _bev_origin: Waypoint

    def __init__(self, og_world_width: float, og_world_height: float, og_width: int, og_height: int) -> None:
        self._virtual_to_real_x_ratio = og_world_width / og_width
        self._virtual_to_real_y_ratio = -og_world_height / og_height
        self._bev_origin = Waypoint(int(og_width/2), int(og_height/2))
    
    def convert_to_world_pose(self, bev_self_location: VehiclePose, target : Waypoint, heading: float = 0) -> VehiclePose:
        dx = target.x - self._bev_origin.x
        dz = target.z - self._bev_origin.z

        dxg = dx * self._virtual_to_real_x_ratio
        dyg = dz * self._virtual_to_real_y_ratio

        return VehiclePose(
            bev_self_location.x + dxg,
            bev_self_location.y + dyg,
            heading)
    
    def convert_to_waypoint(self, bev_self_location: VehiclePose, target: VehiclePose) -> Waypoint:
        dxg = target.x - bev_self_location.x
        dyg = target.y - bev_self_location.y

        dx = dxg / self._virtual_to_real_x_ratio
        dz = dyg / self._virtual_to_real_y_ratio

        return Waypoint (self._bev_origin.x + dx, self._bev_origin.z + dz)
