import sys
from typing import List
from carlasim.carla_sim_controller import CarlaSimulatorController
import time
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')      
from gi.repository import Gst, GLib

Gst.init(None)
simulation = CarlaSimulatorController()
simulation.run()

while True:
    time.sleep(10)