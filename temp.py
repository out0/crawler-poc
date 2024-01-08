import sys, time
sys.path.append("..")
from motion.motion_controller import MotionController
from carlasim.carla_client import CarlaClient
from carlasim.ego_car import EgoCar
from planner.stub_slam import StubSLAM
from planner.stub_global_planner import StubGlobalPlanner

import carla

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

StartPosition = ['Town07', -100, 0, 0, 0.0]
global_planner = StubGlobalPlanner()
global_planner.read_mission("./mission.dat")
client = CarlaClient(town=StartPosition[0])
ego = EgoCar(client)
ego.set_pose(StartPosition[1], StartPosition[2], StartPosition[3], StartPosition[4])
ego.set_power(0)
ego.set_steering(0)

from carlasim.mqtt_client import MqttClient
from carlasim.ego_car_remote_controller import EgoCarRemoteController
mqtt_client = MqttClient("10.0.1.5", 1883)
manual_control = EgoCarRemoteController(mqtt_client, ego)

world = client.get_world()

for w in global_planner.get_all_poses():    
    world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
    
def set_sterr(angle: float) -> None:
    ego.set_steering(angle)
    pass

def invalid_path() -> None:
    print("motion has reached an invalid path")

slam = StubSLAM(ego)

ctrl = MotionController (
    period_ms=100,
    on_invalid_path=invalid_path, 
    odometer=lambda : ego.odometer.read(),
    power_actuator=lambda p: ego.set_power(p),
    brake_actuator=lambda p: ego.set_brake(p),
    steering_actuator=lambda a : set_sterr(a),
    slam_find_current_pose=lambda : slam.estimate_ego_pose()
)

ctrl.set_path(global_planner.get_all_poses())

ctrl.start()

input()

ctrl.destroy()