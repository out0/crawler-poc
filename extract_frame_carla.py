#! /usr/bin/python3
import sys
from typing import List
from carlasim.carla_client import CarlaClient
from carlasim.carla_camera import CarlaCamera
from bev.bev_projector import BEVProjector
from carlasim.video_streamer import VideoStreamer
from carlasim.frame_segment_converter import FrameSegmentConverter
import cv2
import numpy as np
import random
import time

# class FrameCaptureCounter:
#     frame_capture_count: int

#     def __init__(self) -> None:
#         self.frame_capture_count = 0
        
#     def inc(self) -> None:
#         self.frame_capture_count = self.frame_capture_count + 1

#     def block_wait(self) -> None:
#         while self.frame_capture_count < 2:
#             time.sleep(0.01)

# counter = FrameCaptureCounter()

class FrameSelection:
    selected_rgb_frame: any
    selected_seg_frame: any
    _can_replace: bool

    def __init__(self) -> None:
        self.selected_rgb_frame = None
        self.selected_seg_frame = None
        self._can_replace = True
    
    def set_rgb(self, frame) -> None:
        if self._can_replace:
            self.selected_rgb_frame = frame

    def set_seg(self, frame) -> None:
        if self._can_replace:
            self.selected_seg_frame = frame

    def lock_replacement(self) -> None:
        self._can_replace = False

fs = FrameSelection()

def extract_frames(town: str, wait_time_ms: int) -> None:
    client = CarlaClient(town)
    bp = client.get_blueprint("vehicle.tesla.model3")
    bp.set_attribute('color', '63, 183, 183')
    location = random.choice(client.get_world().get_map().get_spawn_points())
    ego_car = client.get_world().spawn_actor(bp, location)
    location = client.get_current_location()
    ego_car.set_location(location)
    ego_car.set_autopilot(True)

    ego_car_rgb_cam = CarlaCamera(client, ego_car, 'sensor.camera.rgb', 800, 600, 120, 60, bev=False)
    ego_car_seg_cam = CarlaCamera(client, ego_car, 'sensor.camera.semantic_segmentation', 400, 300, 120, 60, bev=True)

    print("waiting")

    ego_car_rgb_cam.set_on_frame_callback(select_rgb_frame)
    ego_car_seg_cam.set_on_frame_callback(proc_seg_frame)

    time.sleep(wait_time_ms/1000)

    print("start capturing")
    fs.lock_replacement()

    selected_rgb_frame = VideoStreamer.to_rgb_array(fs.selected_rgb_frame)
    cv2.imwrite('original.png', selected_rgb_frame)

    selected_seg_frame = VideoStreamer.to_rgb_array(fs.selected_seg_frame)
    converter = FrameSegmentConverter(400, 300)
    cv2.imwrite('segmented.png', converter.convert_clone_frame(selected_seg_frame))

    # projector = BEVProjector()
    # selected_seg_frame = projector(selected_seg_frame)
    # converter = FrameSegmentConverter(720, 960)
    # converter.convert_frame(selected_seg_frame)
    # cv2.imwrite('bev_segmented.png', selected_seg_frame)

    print("done")

def select_rgb_frame(frame) -> None:
    fs.set_rgb(frame)

def proc_seg_frame(frame) -> None:
    fs.set_seg(frame)


def _get_wait_time(argc: int, argv: List[str]) -> int:
    if argc > 1:
        try:
            return int(argv[1])
        finally:
            pass
    return 1000


def main(argc: int, argv: List[str]) -> int:
    wait_time_ms = _get_wait_time(argc, argv)
    extract_frames('Town07', wait_time_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
