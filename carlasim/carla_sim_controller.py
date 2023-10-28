import random
from .carla_client import CarlaClient
from .video_streamer import VideoStreamer
from .carla_camera import CarlaCamera
from bev.bev_projector import BEVProjector
import threading
from typing import Any
import numpy as np
from utils.func_timing import time_exec
from .frame_segment_converter import FrameSegmentConverter
from .vehicle_hal import EgoCar
from planner.vehicle_pose import VehiclePose

class CarlaSimulatorController:
    _carla_client: CarlaClient
    _car: EgoCar
    _frame_to_plan: any
    _bev_to_stream: any
    _is_running: bool
    __copying_frame: bool

    def __init__(self) -> None:
        self._carla_client = None
        self._waiting_for_proc_frames = True
        self._current_frame = None
        self._bev_to_stream = None
        self._on_new_bev_frame_callback = None
        self.__copying_frame = False
        
       
    def start(self, town: str, first_pose: VehiclePose) -> None:
        self._carla_client = CarlaClient(town)

        self._car = EgoCar(self._carla_client)\
            .with_rgb_camera()\
            .with_bev_camera(on_frame_callback=self._on_segmented_frame)\
            .build()
        
        self._car.set_pose(first_pose.x, first_pose.y, first_pose.heading)

    def _on_segmented_frame(self, frame):
        if self.__copying_frame:
            return
        
        self._current_frame = frame
        self._car.send_bev_frame(self._bev_to_stream)

    def _set_pixel_color(self, frame, y: int, x: int, r: int, g: int, b: int):
        frame[y][x][0] = r
        frame[y][x][1] = g
        frame[y][x][2] = b


    def get_vehicle(self):
        return self._car
          
    def get_frame(self) -> np.ndarray:
        self.__copying_frame = True
        frame = VideoStreamer.to_rgb_array(self._current_frame)
        self.__copying_frame = False
        return frame

    def terminate(self) -> None:
        self._car.destroy()
