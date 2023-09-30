#! /usr/bin/python3
import cv2
import sys
import numpy as np
from bev.bev_projector import BEVProjector

from segmentation.frame_segmenter \
    import FrameSegmenter

from segmentation.frame_segmenter_onnx_runtime\
    import FrameSegmenterOnnxRuntimeImpl

from utils.func_timing import time_exec
from utils.video_output import OutputWriter
from planner.astar import AStarPlanner
from planner.waypoint import Waypoint
from typing import List
from threading import Thread
import time

class FrameSelection:
    selected_rgb_frame: any
    selected_seg_frame: any

    def __init__(self) -> None:
        self.selected_rgb_frame = None
        self.selected_seg_frame = None

    def set_rgb(self, frame) -> None:
        self.selected_rgb_frame = frame

    def set_seg(self, frame) -> None:
        self.selected_seg_frame = frame

fs = FrameSelection()


class Timeout:
    _timeout: int
    _thr: Thread
    _is_timed_out: bool

    def __init__(self, timeout: int) -> None:
        self._timeout = timeout
        self._thr = Thread(None, self._run_timeout)
        self._is_timed_out = False
        self._thr.start()
    
    def _run_timeout(self):
        self._is_timed_out = False
        time.sleep(self._timeout / 1000)
        self._is_timed_out = True


    def is_timeout(self) -> bool:
        return self._is_timed_out


def extract_frames(wait_time_ms: int) -> None:
    model = "/workspaces/crawler-poc/assets/rtkbosque.onnx"
    source = "/workspaces/crawler-poc/dataset.mp4"

    input = cv2.VideoCapture(source)
    segmenter: FrameSegmenter = FrameSegmenterOnnxRuntimeImpl(model)
    projector = BEVProjector()

    print("waiting")
    tm = Timeout(wait_time_ms)

    selected_rgb_frame = None
    while (not tm.is_timeout()):
        _, selected_rgb_frame = input.read()

    while selected_rgb_frame is None:
        _, selected_rgb_frame = input.read()

    print("start capturing")
    cv2.imwrite('original.png', selected_rgb_frame)
    
    segmented_frame = segmenter.segment(selected_rgb_frame)
    cv2.imwrite('segmented.png', segmented_frame)
    
    bev_frame = projector(segmented_frame)
    cv2.imwrite('bev_segmented.png', bev_frame)
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
    extract_frames(wait_time_ms)
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
