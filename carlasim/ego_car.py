from .carla_client import CarlaClient
import random
import carla
from carlasim.sensors.carla_camera import FrontSemanticCamera, BEVSemanticCamera
from carlasim.sensors.data_sensors import CarlaGps, GpsData, CarlaIMU, IMUData, CarlaOdometer, OdometerData
import math, threading, time
import numpy as np
from carlasim.video_streamer import VideoStreamer
from carlasim.mqtt_client import MqttClient

GPS_PERIOD_ms = 50
IMU_PERIOD_ms = 50



class PeriodicCaller:
    _period: float
    _call_thread: threading.Thread
    _running: bool
    _method: callable
    _payload: any
    
    def __init__(self, period_ms: int, method: callable, payload: any) -> None:
        self._period = period_ms / 1000
        self._running = False
        self._method = method
        self._call_thread = threading.Thread(target=self._run)
        self._call_thread.start()
        self._payload = payload
    
    def _run(self) -> None:
        self._running = True
        
        while (self._running):
            time.sleep(self._period)
            if self._running:
                self._method(self._payload)

    def destroy(self):
        self._running = False
        
    
class EgoCar:
    _ego_car: any
    _vehicle_control: carla.VehicleControl
    _client: CarlaClient
    
    front_camera: FrontSemanticCamera
    bev_camera: BEVSemanticCamera
    gps: CarlaGps
    imu: CarlaIMU
    odometer: CarlaOdometer
    _video_streamer_front_camera: VideoStreamer
    _video_streamer_bev: VideoStreamer
    _video_streamer_front_camera_periodic_sender: PeriodicCaller
    _video_streamer_bev_periodic_sender: PeriodicCaller
    
    def __init__(self, client: CarlaClient) -> None:
        self._client = client
        bp = self._client.get_blueprint("vehicle.tesla.model3")
        bp.set_attribute('color', '63, 183, 183')
        location = random.choice(self._client.get_world().get_map().get_spawn_points())
        self._ego_car = self._client.get_world().spawn_actor(bp, location)
        self.gps = CarlaGps(client, self._ego_car, GPS_PERIOD_ms/1000)
        self.imu = CarlaIMU(client, self._ego_car, IMU_PERIOD_ms/1000)
        self.odometer = CarlaOdometer(self._ego_car)
        self.front_camera = FrontSemanticCamera(800, 600, 120, 30)
        self.bev_camera = BEVSemanticCamera(400, 300, 30)
        self._vehicle_control = carla.VehicleControl()
        self.remote_controller = None
        self._autonomous_mode = False
        self._video_streamer_front_camera = None
        self._video_streamer_bev = None
        self._video_streamer_front_camera_periodic_sender = None
        self._video_streamer_bev_periodic_sender = None
        

    def destroy(self) -> None:
        self.gps.destroy()
        self.imu.destroy()
        self.odometer.destroy()
        
        if self.front_camera is not None:
            self.front_camera.destroy()

        if self.bev_camera is not None:
            self.bev_camera.destroy()
        
        if self._video_streamer_front_camera_periodic_sender is not None:
            self._video_streamer_front_camera_periodic_sender.destroy()
        
        if self._video_streamer_bev_periodic_sender is not None:
            self._video_streamer_bev_periodic_sender.destroy()
           
        self._ego_car.destroy()
    
    def get_location(self) -> np.array:
        # """ Returns the vehicle's location on the simulator. This value should be used for debugging only
        # 
        # Returns:
        #     _type_: _description_
        # """
        
        p = self._ego_car.get_location()        
        return np.array([p.x, p.y, p.z])
    
    def get_heading(self) -> float:
        # """Returns the vehicle's heading on the simulator. This value should be used for debugging only
        # Returns:
        #     float: _description_
        # """
        t = self._ego_car.get_transform()
        return t.rotation.yaw
    
    def set_pose(self, x: int, y: int, z: int, heading: float) -> None:
        # """Sets a new pose for the vehicle by teleporting it to the target location
        # Args:
        #     x (int): x coordinate in carla's map
        #     y (int): y coordinate in carla's map
        #     z (int): z coordinate in carla's map
        #     heading (float): car's yaw representing it's heading 
        # """
        t = self._ego_car.get_transform()
        t.location =  carla.libcarla.Location(x, y, z)
        t.rotation.yaw = heading
        self._ego_car.set_transform(t)
        
    def set_autopilot(self, value: bool) -> None:
        # Fallback of the control to the simulator for auto driving. This method should be used for debugging only
        # Args:
        #    value (bool): turns on/off the simulator autopilot
        self._ego_car.set_autopilot(value)
        
    def set_power(self, power: int) -> None:
        self._vehicle_control.throttle = abs(power / 240)
        # self._vehicle_control.reverse = False
        self._vehicle_control.brake = 0.0
        self._ego_car.apply_control(self._vehicle_control)

    def set_brake(self, brake: float) -> None:
        self._vehicle_control.throttle = 0.0
        self._vehicle_control.brake = brake
        self._ego_car.apply_control(self._vehicle_control)

    def set_steering(self, angle: int) -> None:
        self._vehicle_control.steer = angle / 40
        self._ego_car.apply_control(self._vehicle_control)
    
    def set_autonomous_mode_flag(self, val: bool) -> None:
        self._autonomous_mode = val
        
    def chk_autonomous_mode_flag(self) -> bool:
        return self._autonomous_mode
    
    def stream_front_camera_to(self, host: str, port: int, width: int = 400, height: int = 300, fps: int = -1) -> None:
        if fps <= 0:
            fps = int(self.front_camera.fps() / 2)
        self._video_streamer_front_camera = VideoStreamer(self.front_camera.width(), self.front_camera.height(), width, height, fps, host, port)
        self._video_streamer_front_camera_periodic_sender = PeriodicCaller(int(1000/fps), lambda ego: ego._video_streamer_front_camera.new_frame(ego.front_camera.read()), self)
        self._video_streamer_front_camera.start()

    def stream_bev_camera_to(self, host: str, port: int, width: int = 400, height: int = 300, fps: int = -1) -> None:
        if fps <= 0:
            fps = int(self.bev_camera.fps() / 2)
        self._video_streamer_bev = VideoStreamer(self.bev_camera.width(), self.bev_camera.height(), width, height, fps, host, port)
        self._video_streamer_bev_periodic_sender = PeriodicCaller(int(1000/fps), lambda ego: ego._video_streamer_bev.new_frame(ego.bev_camera.read()), self)
        self._video_streamer_bev.start()
    
    def stop_stream_front_camera(self) -> None:
        if  self._video_streamer_front_camera is None:
            return
        self._video_streamer_front_camera_periodic_sender.destroy()
        self._video_streamer_front_camera_periodic_sender = None
        self._video_streamer_front_camera.stop()
        self._video_streamer_front_camera = None
    
    def stop_stream_bev_camera(self) -> None:
        if  self._video_streamer_bev is None:
            return
        self._video_streamer_bev_periodic_sender.destroy()
        self._video_streamer_bev_periodic_sender = None
        self._video_streamer_bev.stop()
        self._video_streamer_bev = None

    def get_carla_ego_car_obj(self):
        return self._ego_car