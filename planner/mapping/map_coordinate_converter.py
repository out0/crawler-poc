from planner.waypoint import Waypoint
from planner.vehicle_pose import VehiclePose
import numpy as np

class MapCoordinateConverter:

    coord_conversion_mat : np.ndarray

    def convert_to_world_pose(self, self_location: VehiclePose, target : Waypoint, heading: float = 0) -> VehiclePose:
        pass
    
    def convert_to_waypoint(self, self_location: VehiclePose, target: VehiclePose) -> Waypoint:
        pass

