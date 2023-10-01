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
    _converter: FrameSegmentConverter
    _frame_to_plan: any
    _bev_to_stream: any

    def __init__(self) -> None:
        self._carla_client = CarlaClient('Town07')
        self._waiting_for_proc_frames = True
        self._projector = BEVProjector()
        self._frame_to_plan = None
        self._bev_to_stream = None
        self._proc_thr = threading.Thread(None, self._process_segmented_frame)
        self._proc_thr.start()
        self._converter = FrameSegmentConverter(400, 400)

    def _stream_camera(self, camera: CarlaCamera, port: int) -> VideoStreamer:
        streamer = VideoStreamer(
            camera.width(), camera.height(), 400, 300, 30, '127.0.0.1', port)
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
            self._carla_client, self._ego_car, 'sensor.camera.rgb', 400, 300, 120, 30, bev=False)

        self._stream_camera(self._ego_car_rgb_cam, 20000)

        self._ego_car_seg_cam = CarlaCamera(
            self._carla_client, self._ego_car, 'sensor.camera.semantic_segmentation', 400, 400, 120, 30, bev=True)

        self._video_streamer_bev = VideoStreamer(
            400, 400, 400, 400, 30, '127.0.0.1', 20001)
        self._video_streamer_bev.start()

        self._ego_car_seg_cam.set_on_frame_callback(self._on_segmented_frame)

    def _on_segmented_frame(self, frame):
        if self._frame_to_plan is None:
            self._frame_to_plan = frame

        if not self._bev_to_stream is None:
            self._video_streamer_bev.new_frame(self._bev_to_stream)

    def _set_pixel_color(self, frame, y: int, x: int, r: int, g: int, b: int):
        frame[y][x][0] = r
        frame[y][x][1] = g
        frame[y][x][2] = b

    def _process_segmented_frame(self) -> None:
        while True:
            if self._frame_to_plan is None:
                continue

            frame = time_exec(lambda: VideoStreamer.to_rgb_array(
                self._frame_to_plan), 'to rgb')

            self._bev_to_stream = frame
            self._on_new_bev_frame_callback(frame)

    def run(self, on_new_bev_frame_callback: callable) -> None:
        self._build_egocar()
        self._on_new_bev_frame_callback = on_new_bev_frame_callback
