
import cv2
import sys
from bev.bev_projector import BEVProjector
from planner.astar import AStarPlanner
from planner.waypoint import Waypoint
from segmentation.frame_segmenter import FrameSegmenter
from segmentation.frame_segmenter_onnx_runtime \
    import FrameSegmenterOnnxRuntimeImpl

import os
from utils.func_timing import time_exec
from typing import List


def main(argc: int, argv: List[str]) -> int:
    args = argv[1:]

    model = args[0] if argc > 1 else "/workspaces/crawler-poc-meu/assets/rtkbosque.onnx"
    source = args[1] if argc > 2 else "/workspaces/crawler-poc-meu/dataset.mp4"

    input = cv2.VideoCapture(source)
    segmenter: FrameSegmenter = time_exec(
        lambda: FrameSegmenterOnnxRuntimeImpl(model), 'segmenter load')
    projector = BEVProjector()

    _, frame = input.read()
    if frame is None:
        print("Failed\n")
        exit(1)

    segmented_frame = time_exec(
        lambda: segmenter.segment(frame), 'frame segmentation')

    bev_segmented_frame = time_exec(
        lambda: projector(segmented_frame), 'bev segmented')
    bev_frame = time_exec(lambda: projector(frame), 'bev frame')

    cv2.imwrite('original.png', frame)
    cv2.imwrite('original_bev.png', bev_frame)
    cv2.imwrite('segmented.png', segmented_frame)
    cv2.imwrite('segmented_bev.png', bev_segmented_frame)

    planner = AStarPlanner()

    start = Waypoint(320, bev_frame.shape[1])
    goal = Waypoint(320, 0)

    path = time_exec(lambda: planner.plan(
        bev_frame, start, goal), 'path planning')

    if path is None:
        print("no path was found")
    else:
        print("path found")

        for p in path:
            for k in range(-10, 10):
                bev_frame[p.z][p.x + k][0] = 255
                bev_frame[p.z][p.x + k][1] = 255
                bev_frame[p.z][p.x + k][2] = 255

    cv2.imwrite('segmented_bev_planned.png', bev_frame)
    return 0


if __name__ == "__main__":
    sys.exit(main(len(sys.argv), sys.argv))
