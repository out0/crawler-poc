import sys, time
sys.path.append("..")
from motion.longitudinal_controller import LongitudinalController
from carlasim.carla_client import CarlaClient
from carlasim.ego_car import EgoCar

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)


StartPosition = ['Town07', -100, 0, 0, 0.0]
DesiredStableSpeed = 20

client = CarlaClient(town=StartPosition[0])
ego = EgoCar(client)
ego.set_pose(StartPosition[1], StartPosition[2], StartPosition[3], StartPosition[4])
ego.set_power(0)
ego.set_steering(0)

controller = LongitudinalController(250, 
                                    odometer=lambda : ego.odometer.read(), 
                                    power_actuator=lambda power: ego.set_power(power),
                                    brake_actuator=lambda brake: ego.set_brake(brake))

controller.set_speed(20.0)
controller.start()

while True:
    time.sleep(1)