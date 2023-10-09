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

class CarlaSimulatorController:
    _carla_client: CarlaClient
    _car: EgoCar
    _proc_thr: threading.Thread
    _converter: FrameSegmentConverter
    _frame_to_plan: any
    _bev_to_stream: any
    _bev_count: int

    def __init__(self) -> None:
        self._carla_client = CarlaClient('Town07')
        self._waiting_for_proc_frames = True
        self._frame_to_plan = None
        self._bev_to_stream = None
        self._proc_thr = threading.Thread(None, self._process_segmented_frame)
        self._proc_thr.start()
        self._converter = FrameSegmentConverter(400, 400)
        self._bev_count = 0        
        self._on_new_bev_frame_callback = None
        
        self._car = EgoCar(self._carla_client)\
            .with_rgb_camera()\
            .with_bev_camera(on_frame_callback=self._on_segmented_frame)\
            .build()

    def _on_segmented_frame(self, frame):
        if self._frame_to_plan is None:
            self._frame_to_plan = frame

        self._car.send_bev_frame(self._bev_to_stream)

    def _set_pixel_color(self, frame, y: int, x: int, r: int, g: int, b: int):
        frame[y][x][0] = r
        frame[y][x][1] = g
        frame[y][x][2] = b

    def _process_segmented_frame(self) -> None:
        while True:
            if self._frame_to_plan is None or self._on_new_bev_frame_callback is None:
                continue

            frame = time_exec(lambda: VideoStreamer.to_rgb_array(
                self._frame_to_plan), 'to rgb')

            self._bev_to_stream = self._on_new_bev_frame_callback(frame)
            
            self._bev_count += 1
            self._frame_to_plan = None

    def get_vehicle(self):
        return self._car
          

    def run(self, on_new_bev_frame_callback: callable) -> None:
        self._on_new_bev_frame_callback = on_new_bev_frame_callback
