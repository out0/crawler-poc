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

class CarlaSimulatorController:
    _carla_client: CarlaClient
    _ego_car: Any
    _ego_car_rgb_cam: Any
    _ego_car_seg_cam: Any
    _video_streamer_bev: VideoStreamer
    _projector: BEVProjector
    _proc_thr: threading.Thread
    _bev_frame_being_processed: any
    _bev_last_frame_processed: any
    _converter: FrameSegmentConverter

    def __init__(self) -> None:
        self._carla_client = CarlaClient('Town07')
        self._waiting_for_proc_frames = True
        self._projector = BEVProjector()
        self._bev_frame_being_processed = None
        self._bev_last_frame_processed = None
        self._proc_thr = threading.Thread(None, self._process_segmented_frame)
        self._proc_thr.start()
        self._converter = FrameSegmentConverter()

    def _stream_camera(self, camera: CarlaCamera, port: int) -> VideoStreamer:
        streamer = VideoStreamer(
            camera.width(), camera.height(), 800, 600, 30, '127.0.0.1', port)
        streamer.start()
        camera.set_on_frame_callback(streamer.new_frame)

    def _build_egocar(self) -> None:
        bp = self._carla_client.get_blueprint("vehicle.tesla.model3")
        bp.set_attribute('color', '63, 183, 183')

        location = random.choice(
            self._carla_client.get_world().get_map().get_spawn_points())
        self._ego_car = self._carla_client.get_world().spawn_actor(bp, location)
        location = self._carla_client.get_current_location()
        self._ego_car.set_location(location)
        self._ego_car.set_autopilot(True)

        self._ego_car_rgb_cam = CarlaCamera(
            self._carla_client, self._ego_car, 'sensor.camera.rgb', 800, 600, 120, 60)

        # self._stream_camera(self._ego_car_rgb_cam, 20000)

        self._ego_car_seg_cam = CarlaCamera(
            self._carla_client, self._ego_car, 'sensor.camera.semantic_segmentation', 1920, 1440, 120, 60)

        self._video_streamer_bev = VideoStreamer(
            720, 960, 720, 960, 30, '127.0.0.1', 20000)
        self._video_streamer_bev.start()

        self._ego_car_seg_cam.set_on_frame_callback(self._on_segmented_frame)

    def _on_segmented_frame(self, frame):       
        if self._bev_frame_being_processed is None:
            self._bev_frame_being_processed = frame

        if self._bev_last_frame_processed is None:
            return

        self._video_streamer_bev.new_frame(self._bev_last_frame_processed)

    def _set_pixel_color(self, frame, y: int, x: int, r: int, g: int, b: int):
        frame[y][x][0] = r
        frame[y][x][1] = g
        frame[y][x][2] = b

    def _process_segmented_frame(self) -> None:
        while True:
            if self._bev_frame_being_processed is None:
                continue

            f = time_exec(lambda: VideoStreamer.to_rgb_array(self._bev_frame_being_processed), 'to rgb' )            
            f = time_exec(lambda: self._projector(f), 'compute bev')
            time_exec(lambda: self._converter.convert_frame(f), 'color convert')
            self._bev_last_frame_processed = f            
            self._bev_frame_being_processed = None
            

    def run(self) -> None:
        self._build_egocar()
