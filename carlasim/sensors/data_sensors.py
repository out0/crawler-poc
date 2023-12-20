from planner.waypoint import Waypoint
from carlasim.carla_client import CarlaClient
import carla
import threading
import math


class PeriodicDataSensor:
    _sensor: any
    _last_sensor_data: any
    _lock: threading.Lock    
    
    def __init__(self, bp: str, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        sensor_bp = client.get_blueprint(bp)
        sensor_bp.set_attribute('sensor_tick', str(capture_period_in_seconds))
        location = carla.Location(x=1.5, y=0.0, z=1)
        rotation = carla.Rotation()
        transform = carla.Transform(location, rotation)
        self._sensor = client.get_world().spawn_actor(sensor_bp, transform, attach_to=vehicle)
        self._sensor.listen(self.__new_data)
        self._lock = threading.Lock()
        self._last_sensor_data = None

    def destroy(self) -> None:
        if self._sensor is None:
            return
        self._sensor.destroy()
        self._sensor = None
        
    def __new_data(self, sensor_data: any):
        if self._lock.acquire(blocking=False):
            self._last_sensor_data = sensor_data
            self._lock.release()
    
    def read(self) -> any:
        while self._sensor is not None:
            if self._lock.acquire(blocking=True, timeout=0.5):
                f = self._last_sensor_data
                self._lock.release()
                return f
        return None

class GpsData:
    latitude: float
    longitude: float
    altitude: float
    
    def __init__(self, lat: float, lon: float, alt: float) -> None:
        self.latitude = lat
        self.longitude = lon
        self.altitude = alt

class CarlaGps (PeriodicDataSensor):    
    def __init__(self, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        super().__init__("sensor.other.gnss", client, vehicle, capture_period_in_seconds)

    def read(self) -> GpsData:
        carla_gps = super().read()
        return GpsData(carla_gps.latitude, carla_gps.longitude, carla_gps.altitude)
        
class IMUData:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    heading: float
    
    def __init__(self) -> None:
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.heading = 0.0

class CarlaIMU (PeriodicDataSensor):    
    def __init__(self, client: CarlaClient, vehicle: any, capture_period_in_seconds: float) -> None:
        super().__init__("sensor.other.imu", client, vehicle, capture_period_in_seconds)

    def read(self) -> IMUData:
        carla_imu = super().read()
        data = IMUData()
        data.accel_x = carla_imu.accelerometer[0]
        data.accel_y = carla_imu.accelerometer[1]
        data.accel_z = carla_imu.accelerometer[2]
        data.heading = carla_imu.compass
        data.gyro_x = carla_imu.gyroscope [0]
        data.gyro_y = carla_imu.gyroscope [1]
        data.gyro_z = carla_imu.gyroscope [2]        
        return data

class OdometerData:
    vel_x: float
    vel_y: float
    vel_z: float
    
    def __init__(self) -> None:
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0


class CarlaOdometer:
    
    _vehicle: any
    
    def __init__(self, vehicle: any) -> None:
        self._vehicle = vehicle


    def read(self) -> float:
        velocity = self._vehicle.get_velocity()
        return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2)
    
    def destroy(self) -> None:
        pass