class TeleportMotionPlanner:
    _car: EgoCar
    _map_converter: MapCoordinateConverterCarla
    
    def __init__(self, car: EgoCar, og_width: int, og_height: int) -> None:
        self._car = car
        self._map_converter = MapCoordinateConverterCarla(36, 30, og_width, og_height)

    def move_to(self, location: VehiclePose, goal: Waypoint) -> None:
        pose = self._map_converter.convert_to_world_pose(location, goal)
        self._car.set_pose(pose.x, pose.y, pose.heading)

    def stop(self) -> None:
        self._car.stop()
