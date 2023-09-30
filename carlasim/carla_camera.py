import carla
from .carla_client import CarlaClient

class CarlaCamera:
    _camera: any
    _carla_client: any
    _width: int
    _height: int
    _fps: int

    def __init__(self, client: CarlaClient, vehicle: any, camera_type:str, width: int, height: int, fov: float, fps: int) -> None:
        self._carla_client = client     
        self._width = width
        self._height = height
        self._fps = fps
        self._camera = self._attach_camera(vehicle, camera_type, width, height, fov, fps)

    def _attach_camera(self, target, camera_type, width: int, height: int, fov: float, fps: int) -> None:
        camera_bp = self._carla_client.get_blueprint(camera_type)
        camera_bp.set_attribute('image_size_x', str(width))
        camera_bp.set_attribute('image_size_y', str(height))
        camera_bp.set_attribute('fov', str(fov))
        camera_bp.set_attribute('sensor_tick', str(1/fps))
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2))
        return self._carla_client.get_world().spawn_actor(camera_bp, camera_transform, attach_to=target)

    def width(self):
        return self._width

    def height(self):
        return self._height

    def fps(self):
        return self._fps

    def set_on_frame_callback(self, callback: callable) -> None:
        self._camera.listen(callback)
        
