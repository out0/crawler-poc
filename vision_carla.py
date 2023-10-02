import sys
from typing import List
from carlasim.carla_sim_controller import CarlaSimulatorController
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')      
from gi.repository import Gst, GLib
from planner.astar import AStarPlanner, Waypoint
from carlasim.video_streamer import VideoStreamer
from carlasim.frame_segment_converter import FrameSegmentConverter
Gst.init(None)

from threading import Thread
import time

class LocalPathPlanner:
    _ego_waypoint: Waypoint
    _goal_waypoint: Waypoint
    _local_planner: AStarPlanner 
    _video_streamer_bev: VideoStreamer
    _last_planned_frame: any
    _continuous_stream_thr: Thread
    _converter: FrameSegmentConverter

    def __init__(self) -> None:
        self._local_planner = AStarPlanner()
        self._ego_waypoint = Waypoint(200, 140)
        self._goal_waypoint = Waypoint(200, 300)
        self._video_streamer_bev = VideoStreamer(400, 300, 400, 300, 30, '127.0.0.1', 20003)
        self._video_streamer_bev.start()
        self._last_planned_frame = None
        self._continuous_stream_thr = Thread(None,  self._transmit_last_planned_frame)
        self._continuous_stream_thr.start()
        self._converter = FrameSegmentConverter(400, 300)

    def _transmit_last_planned_frame(self) -> None:
        while True:
            if self._last_planned_frame is None:
                continue

            self._video_streamer_bev.new_frame(self._last_planned_frame)
            time.sleep(1/30)

    def _transmit_planned_path(self, bev, path: List[Waypoint]):

        frame = self._converter.convert_clone_frame(bev)

        if not (path is None):
            for w in path:
                frame[w.z][w.x][0] = 255
                frame[w.z][w.x][1] = 255
                frame[w.z][w.x][2] = 255
        
        self._last_planned_frame = frame     

    def on_new_frame(self, bev) -> None:
        path = self._local_planner.plan(bev, self._ego_waypoint, self._goal_waypoint)
        self._transmit_planned_path(bev, path)

planner = LocalPathPlanner()
simulation = CarlaSimulatorController()
simulation.run(planner.on_new_frame)

while True:
    time.sleep(10)