import carla

class CarlaActor:
    _actor_obj: any

    def __init__(self, actor_obj) -> None:
        self._actor_obj = actor_obj

    def set_actor_obj(self, actor_obj) -> None:
        self._actor_obj = actor_obj

    def set_location(self, x: int, y: int) -> None:
        t = self._actor_obj.get_transform()
        t.location =  carla.libcarla.Location(x, y, 0)
        self._actor_obj.set_transform(t)

    def set_pitch(self, pitch: float) -> None:
        t = self._actor_obj.get_transform()
        t.rotation.pitch = pitch
        self._actor_obj.set_transform(t)
