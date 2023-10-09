from planner.waypoint import Waypoint
from carlasim.carla_sim_controller import CarlaSimulatorController

class CarlaGps:
    _carla_controller: CarlaSimulatorController
    def __init__(self, carla_controller: CarlaSimulatorController) -> None:
        self._carla_controller = carla_controller

    def read(self) -> Waypoint:
        location = self._carla_controller.get_vehicle().get_location()
        print(type(location))
        return Waypoint(location.x, location.y)
