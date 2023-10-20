#
#  This class provide simple methods for controlling the simulated car 
#

from .carla_client import CarlaClient
import random
import carla
from .carla_camera import CarlaCamera
from .video_streamer import VideoStreamer
import math


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
    _color: str


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
        self._color = '63, 183, 183'

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
    
    def set_color(self, r: int, g: int, b: int):
        self._color = f'{r}, {g}, {b}'
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
        bp.set_attribute('color', self._color)

        if self._spawn_location is None:
            location = random.choice(self._client.get_world().get_map().get_spawn_points())
            self._ego_car = self._client.get_world().spawn_actor(bp, location)
            location = self._client.get_current_location()
            self._ego_car.set_location(location)
        else:
            self._ego_car = self._client.get_world().spawn_actor(bp, self._spawn_location)

        if self._config_autopilot:
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
    
    def get_heading(self):
        t = self._ego_car.get_transform()
        return t.rotation.yaw

        # if y < 0:
        #     return 360 + y
        # return y
    
    def set_pose(self, x: int, y: int, yaw: float) -> None:
        self.stop()
        t = self._ego_car.get_transform()
        t.location =  carla.libcarla.Location(x, y, 0)
        t.rotation.yaw = yaw
        self._ego_car.set_transform(t)
        

    def get_actor(self):
        return self._ego_car

    def _set_engine_power(self, power:int):        
        self._vehicle_control.throttle = abs(power) / 240
        self._vehicle_control.reverse = power < 0
        self._vehicle_control.brake = 0.0
        self._ego_car.apply_control(self._vehicle_control)       

    def stop(self):
        self._vehicle_control.brake = 1.0
        self._ego_car.apply_control(self._vehicle_control)   

    def forward(self):
        self._set_engine_power(100)

    def forward_slow(self):
        self._set_engine_power(50)

    def backward(self):
        self._set_engine_power(-100)
    
    def steer(self, angle: int):
        self._vehicle_control.steer = angle / 40
        self._ego_car.apply_control(self._vehicle_control)

    def set_autopilot(self, val: bool):
        """ Simulator-only """
        self._ego_car.set_autopilot(val)

    def get_yaw(self):
        """ Simulator-only - needs SLAM / RESEARCH """
        return self._ego_car.get_transform().rotation.yaw

    # def _calculate_angle(self, x1, y1, x2, y2):
    #     angle_rad = math.atan2(y2 - y1, x2 - x1)
    #     angle_deg = math.degrees(angle_rad)    
    #     return angle_deg

    # def compute_heading_to(self, x2, y2):
    #     current_location = self.get_location()
    #     return self._calculate_angle(current_location.x, current_location.y, x2, y2)       

    # def compute_distance_to(self, x2, y2) -> float:
    #     current_location = self.get_location()
    #     return math.sqrt((y2 - current_location.y)**2 + (x2 - current_location.x)**2)


    def drive_to(self, x_goal:int, y_goal:int) -> None:
        new_heading = self.compute_heading_to(x_goal, y_goal)
        # self.steer(1.5 * (new_heading - self.get_heading()))
        # self.stop()
        # while int(self.get_heading()) != int(new_heading):
        #     self.forward_slow()
        # self.steer(0)
        # last_d = self.compute_distance_to(x_goal, y_goal)
        # self.forward()
        # d = last_d
        # while d <= last_d:
        #     d = self.compute_distance_to(x_goal, y_goal)
        # self.stop()
        self.set_pose(x_goal, y_goal, new_heading)
