from model.slam import SLAM
from model.vehicle_pose import VehiclePose
from carlasim.ego_car import EgoCar

class StubSLAM (SLAM):
    _car: EgoCar

    def __init__(self, car: EgoCar) -> None:
        self._car = car
        pass

    def estimate_ego_pose(self) -> VehiclePose:
        location = self._car.get_location()
        heading = self._car.get_heading()
        return VehiclePose(location[0], location[1], heading)
