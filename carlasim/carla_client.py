import carla

class CarlaClient:
    _client: any
    _world: any
    _bplib : any

    def __init__(self, town: str = 'Town03') -> None:
        self._client = carla.Client('localhost', 2000)
        self._client.set_timeout(110.0)
        self._client.load_world(town)
        self._world = self._client.get_world()
        self._bplib = self._world.get_blueprint_library()

    def get_world(self):
        return self._world

    def get_blueprint(self, blueprint_name:str):
        return self._bplib.find(blueprint_name)

    def get_current_location(self):
        return self._world.get_spectator().get_location()

