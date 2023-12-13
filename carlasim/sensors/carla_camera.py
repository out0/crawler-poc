import carla
from typing import List
from carlasim.carla_client import CarlaClient
import threading
import time
from carlasim.sensors.frame_segment_converter_cuda import FrameSegmentConverterCuda 
import numpy as np

class CarlaCamera:
    _camera: any
    _width: int
    _height: int
    _fov: float
    _fps: int
    _location_vector: List[float]
    _camera_type: str
    _last_frame: any
    _lock: threading.Lock
    _last_timestamp: float


    def __init__(self) -> None:
        self._camera = None
        self._width = 800
        self._height = 600
        self._fov = 120
        self._fps = 30
        self._location_vector = [1.5, 0.0, 2.0]
        self._rotation_vector = [0.0, 0.0, 0.0]
        self._camera_type = 'sensor.camera.rgb'
        self._lock = threading.Lock()
        self._last_frame = None

    def set_parameters(self, camera_type: str, width: int, height: int, fov: float, fps: int) -> None:
        self._camera_type = camera_type
        self._width = width
        self._height = height
        self._fov = fov
        self._fps = fps
        
    def set_location(self, x: float, y: float, z: float) -> None:
        self._location_vector = [x, y, z]
        
    def set_rotation(self, pitch: float, yaw: float, roll: float) -> None:
        self._rotation_vector = [pitch, yaw, roll]

    def attach_to(self, client: CarlaClient, vehicle: any) -> None:
        camera_bp = client.get_blueprint(self._camera_type)
        camera_bp.set_attribute('image_size_x', str(self._width))
        camera_bp.set_attribute('image_size_y', str(self._height))
        camera_bp.set_attribute('fov', str(self._fov))
        camera_bp.set_attribute('sensor_tick', str(1/self._fps))
        
        location = carla.Location(x=self._location_vector[0], y=self._location_vector[1], z=self._location_vector[2])
        rotation = carla.Rotation(pitch=self._rotation_vector[0], yaw=self._rotation_vector[1], roll=self._rotation_vector[2])
        
        camera_transform = carla.Transform(location, rotation)
        self._camera = client.get_world().spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        self._camera.listen(self.new_frame)

    def destroy(self) -> None:
        if self._camera is None:
            return
        
        self._camera.destroy()
        self._camera = None

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def fps(self) -> int:
        return self._fps
    
    def new_frame(self, frame):
        if frame is None:
            return
        
        if self._lock.acquire(blocking=False):
            self._last_frame = self._filter_frame(frame)
            self._last_timestamp = time.time()
            self._lock.release()
    
    def _to_bgra_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a RGB numpy array."""
        array = CarlaCamera._to_bgra_array(image)
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    
    def _filter_frame(self, frame):
        return CarlaCamera.to_rgb_array(frame)

    def read(self) -> any:
        while self._camera is not None:
            if self._lock.acquire(blocking=True, timeout=0.5):
                f = self._last_frame
                self._lock.release()
                return f
        return None

class FrontRGBCamera(CarlaCamera):    
    def __init__(self, width: int, height: int, fov: int, fps: int) -> None:
        super().__init__()
        self.set_parameters(camera_type='sensor.camera.rgb',
                                   width=width, height=height, fov=fov, fps=fps)
        self.set_location(x=1.5, y=0.0, z=2)

class FrontSemanticCamera(CarlaCamera):    
    def __init__(self, width: int, height: int, fov: int, fps: int) -> None:
        super().__init__()
        self.set_parameters(camera_type='sensor.camera.semantic_segmentation',
                                   width=width, height=height, fov=fov, fps=fps)
        self.set_location(x=1.5, y=0.0, z=2)

class FrontColoredSemanticCamera (FrontSemanticCamera):
    __frame_segment_converter: FrameSegmentConverterCuda

    def __init__(self, width: int, height: int, fov:int, fps: int) -> None:
        super().__init__(width, height, fov, fps)
        self.__frame_segment_converter = FrameSegmentConverterCuda()

    def _filter_frame(self, frame):
        frame = super()._filter_frame(frame)
        return self.__frame_segment_converter.convert_frame(frame)

class BEVRGBCamera (CarlaCamera):
    def __init__(self, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.set_parameters(camera_type='sensor.camera.rgb',
                                   width=width, height=height, fov=120, fps=fps)
        self.set_location(x=1.5, y=0.0, z=10)
        self.set_rotation(pitch=-90, yaw=0, roll=0)

class BEVSemanticCamera (CarlaCamera):
    def __init__(self, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.set_parameters(camera_type='sensor.camera.semantic_segmentation',
                                   width=width, height=height, fov=120, fps=fps)
        self.set_location(x=1.5, y=0.0, z=10)
        self.set_rotation(pitch=-90, yaw=0, roll=0)

class BEVColoredSemanticCamera (BEVSemanticCamera):
    __frame_segment_converter: FrameSegmentConverterCuda

    def __init__(self, width: int, height: int, fps: int) -> None:
        super().__init__(width, height, fps)
        self.__frame_segment_converter = FrameSegmentConverterCuda()

    def _filter_frame(self, frame):
        frame = super()._filter_frame(frame)
        return self.__frame_segment_converter.convert_frame(frame)
