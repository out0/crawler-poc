import sys, time
sys.path.append("..")
from motion.longitudinal_controller import LongitudinalController
from motion.lateral_controller import LateralController
from carlasim.carla_client import CarlaClient
from carlasim.ego_car import EgoCar
from planner.stub_slam import StubSLAM
from planner.stub_global_planner import StubGlobalPlanner
from model.waypoint import DistWaypoint
import carla

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)


StartPosition = ['Town07', -100, 0, 0, 0.0]
DesiredStableSpeed = 20

global_planner = StubGlobalPlanner()
global_planner.read_mission("./small_mission.dat")

client = CarlaClient(town=StartPosition[0])
ego = EgoCar(client)
ego.set_pose(StartPosition[1], StartPosition[2], StartPosition[3], StartPosition[4])
ego.set_power(0)
ego.set_steering(0)

world = client.get_world()

for w in global_planner.get_all_poses():    
    world.debug.draw_string(carla.Location(w.x, w.y, 2), 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)

acc = LongitudinalController(250, 
                                    odometer=lambda : ego.odometer.read(), 
                                    power_actuator=lambda power: ego.set_power(power),
                                    brake_actuator=lambda brake: ego.set_brake(brake))

acc.set_speed(0.0)


slam = StubSLAM(ego)


def set_sterr (p: float):
    print (f"steering to {p}")
    ego.set_steering(p)

lc = LateralController(slam_find_current_pose=lambda : slam.estimate_ego_pose(), 
                       desired_look_ahead_dist=10.0,
                       odometer=lambda : ego.odometer.read(),
                       set_steering_angle=lambda a : set_sterr(a),
                       period_ms=50)

poses = global_planner.get_all_poses()

acc.start()
lc.start()


