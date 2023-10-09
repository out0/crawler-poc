#
#  This class provide simple methods for controlling the simulated car 
#

from .carla_client import CarlaClient
import random
import carla
from .carla_camera import CarlaCamera
from .video_streamer import VideoStreamer

class EgoCar:
    _ego_car: any
    _client: CarlaClient
    _config_rgb_cam_port: int
    _config_bev_cam_port: int
    _config_rgb_frame_callback: callable
    _config_bev_frame_callback: callable
    _spawn_location: carla.libcarla.Location
    _camera_rgb: CarlaCamera
    _camera_bev: CarlaCamera
    _camera_rgb_streamer: VideoStreamer
    _camera_bev_streamer: VideoStreamer
    _vehicle_control: carla.VehicleControl


    def __init__(self, client: CarlaClient) -> None:
        self._ego_car = None
        self._client = client
        self._config_rgb_cam_port = -1
        self._config_bev_cam_port = -1
        self._config_rgb_frame_callback = None
        self._config_bev_frame_callback = None
        self._config_autopilot = False
        self._spawn_location = None
        self._camera_rgb = None
        self._camera_bev = None
        self._camera_rgb_streamer = None
        self._camera_bev_streamer = None
        self._vehicle_control = carla.VehicleControl()

    def with_rgb_camera(self, port: int = 20000, on_frame_callback: callable = None) -> 'EgoCar':
        self._config_rgb_cam_port = port
        self._config_rgb_frame_callback = on_frame_callback
        return self

    def with_bev_camera(self, port: int = 20001, on_frame_callback: callable = None) -> 'EgoCar':
        self._config_bev_cam_port = port
        self._config_bev_frame_callback = on_frame_callback
        return self

    def with_spawn_location(self, location: carla.libcarla.Location) -> 'EgoCar':
        self._spawn_location = location
        return self

    def autopilot(self) -> 'EgoCar':
        self._config_autopilot = True
        return self
    
    def send_bev_frame(self, frame) -> None:
        if frame is None:
            return
        
        self._build_bev_streamer()
        if self._camera_bev_streamer is None:
            return
        self._camera_bev_streamer.new_frame(frame)

    def send_rgb_frame(self, frame) -> None:
        if frame is None:
            return

        self._build_rgb_streamer()
        if self._camera_rgb_streamer is None:
            return
        self._camera_rgb_streamer.new_frame(frame)

    def _build_bev_streamer(self) -> None:
        if self._camera_bev_streamer is None:
            self._camera_bev_streamer = VideoStreamer(
                self._camera_bev.width(), self._camera_bev.height(), 400, 300, 30, '127.0.0.1', self._config_bev_cam_port)
            self._camera_bev_streamer.start()

    def _build_rgb_streamer(self) -> None:
        if self._camera_rgb_streamer is None:
            self._camera_rgb_streamer = VideoStreamer(
                self._camera_rgb.width(), self._camera_rgb.height(), 400, 300, 30, '127.0.0.1', self._config_rgb_cam_port)
            self._camera_rgb_streamer.start()        

    def build(self) -> 'EgoCar': 
        bp = self._client.get_blueprint("vehicle.tesla.model3")
        bp.set_attribute('color', '63, 183, 183')

        if self._spawn_location is None:
            location = random.choice(self._client.get_world().get_map().get_spawn_points())
            self._ego_car = self._client.get_world().spawn_actor(bp, location)
            location = self._client.get_current_location()
            self._ego_car.set_location(location)
        else:
            self._ego_car = self._client.get_world().spawn_actor(bp, self._spawn_location)

        self._ego_car.set_autopilot(self._config_autopilot)

        if self._config_rgb_cam_port > 0:
            self._camera_rgb = CarlaCamera(self._client, self._ego_car, 'sensor.camera.rgb', 400, 300, 120, 30, bev=False)
            if self._config_rgb_frame_callback is None:
                self._build_rgb_streamer()
                self._camera_rgb.set_on_frame_callback(self._camera_rgb_streamer.new_frame)
            else:
                self._camera_rgb.set_on_frame_callback(self._config_rgb_frame_callback)
        
        if self._config_bev_cam_port > 0:
            self._camera_bev = CarlaCamera(self._client, self._ego_car, 'sensor.camera.semantic_segmentation', 400, 300, 120, 30, bev=True)
            if self._config_bev_frame_callback is None:
                self._build_bev_streamer()
                self._camera_bev.set_on_frame_callback(self._camera_bev_streamer.new_frame)
            else:
                self._camera_bev.set_on_frame_callback(self._config_bev_frame_callback)
        
        return self

    def get_location(self):
        return self._ego_car.get_location()

    def get_actor(self):
        return self._ego_car

    def _set_engine_power(self, power:int):        
        self._vehicle_control.throttle = abs(power) / 240
        self._vehicle_control.reverse = power < 0
        self._vehicle_control.brake = 0.0
        self._ego_car.apply_control(self._vehicle_control)       

    def stop(self):
        self._vehicle_control.brake = 100.0

    def forward(self):
        self._set_engine_power(100)

    def backward(self):
        self._set_engine_power(-100)
    
    def steer(self, angle: int):
        self._vehicle_control.steer = angle / 40
        self._ego_car.apply_control(self._vehicle_control)

    def set_autopilot(self, val: bool):
        self._ego_car.set_autopilot(val)