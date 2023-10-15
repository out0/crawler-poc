
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
from planner.astar import AStarPlanner, PlannerResult
from planner.waypoint import Waypoint

if __name__ == '__main__':
    args = sys.argv[1:]
    model = args[1] if len(
        args) > 1 else "/workspaces/crawler-poc/assets/rtkbosque.onnx"
    source = args[2] if len(
        args) > 2 else "/workspaces/crawler-poc/dataset.mp4"

    reading = True
    input = cv2.VideoCapture(source)
    output = OutputWriter()
    segmenter: FrameSegmenter = FrameSegmenterOnnxRuntimeImpl(model)
    projector = BEVProjector()

    print('writing outputs...')

    f_count = 0

    while f_count < 3000:
        reading, frame = input.read()
        if frame is None:
            continue
        segmented_frame = segmenter.segment(frame)
        bev_frame = projector(segmented_frame)

        planner = AStarPlanner()

        output.on_original_frame(frame)
        output.on_segmented_frame(segmented_frame)

        f_count = f_count + 1

        start = Waypoint(320, bev_frame.shape[1] - 1)
        goal = Waypoint(320, 0)

        result: PlannerResult = time_exec(lambda: planner.plan(bev_frame, start, goal), 'path planning')

        if result.valid:
            print("path found")
            for p in result.path:
                for k in range(-10, 10):
                    bev_frame[p.z][p.x + k][0] = 255
                    bev_frame[p.z][p.x + k][1] = 255
                    bev_frame[p.z][p.x + k][2] = 255

        output.on_planned_frame(bev_frame)
    
    output.close()