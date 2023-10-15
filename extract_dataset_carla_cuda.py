#! /usr/bin/python3
import sys
from typing import List
from carlasim.carla_client import CarlaClient
from carlasim.carla_camera import CarlaCamera
from carlasim.video_streamer import VideoStreamer
from carlasim.frame_segment_converter_cuda import FrameSegmentConverterCuda
import cv2
import random
import time

class Controller:
    outp_orig: cv2.VideoWriter
    outp_bev: cv2.VideoWriter
    outp_bev_color: cv2.VideoWriter
    is_streaming: bool
    frame_segmenter: FrameSegmentConverterCuda

    def __init__(self) -> None:
        self.outp_orig = cv2.VideoWriter('orig.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300))
        self.outp_bev = cv2.VideoWriter('bev_orig.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300))
        self.outp_bev_color = cv2.VideoWriter('bev_color.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300))
        self.frame_segmenter = FrameSegmentConverterCuda()
        self.is_streaming = True

controller = Controller()

def build_video_output(town: str, wait_time_s: int) -> None:
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

    ego_car_rgb_cam.set_on_frame_callback(select_rgb_frame)
    ego_car_seg_cam.set_on_frame_callback(proc_seg_frame)

    print("start capturing")
    time.sleep(wait_time_s)
    print("done")
    controller.is_streaming = False
    
    controller.outp_orig.release()
    controller.outp_bev.release()
    controller.outp_bev_color.release()

def select_rgb_frame(frame) -> None:
    if controller.is_streaming:
        controller.outp_orig.write(VideoStreamer.to_rgb_array(frame))

def proc_seg_frame(frame) -> None:
    if controller.is_streaming:
        f = VideoStreamer.to_rgb_array(frame)
        controller.outp_bev.write(f)
        controller.outp_bev_color.write(controller.frame_segmenter.convert_frame(f))


def _get_wait_time(argc: int, argv: List[str]) -> int:
    if argc > 1:
        try:
            return int(argv[1])
        finally:
            pass
    return 10


def main(argc: int, argv: List[str]) -> int:
    wait_time_ms = _get_wait_time(argc, argv)
    build_video_output('Town07', wait_time_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))