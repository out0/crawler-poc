import sys
from typing import List
from carlasim.carla_sim_controller import CarlaSimulatorController
import time
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')      
from gi.repository import Gst, GLib

Gst.init(None)


class LocalPathPlanner:
    def __init__(self) -> None:
        pass

    def on_new_frame(self, bev) -> None:
        
        pass

planner = LocalPathPlanner()
simulation = CarlaSimulatorController()
simulation.run(planner.on_new_frame)

while True:
    time.sleep(10)