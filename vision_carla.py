import cv2
import math
import numpy as np
import time
from carlasim.frame_segment_converter import FrameSegmentConverter
from planner.astar import AStarPlanner, Waypoint
from gi.repository import Gst, GLib
from typing import List
from carlasim.carla_sim_controller import CarlaSimulatorController
from carlasim.gps import CarlaGps
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
Gst.init(None)
from carlasim.remote_controller import RemoteController
from carlasim.mqtt_client import MqttClient

class LocalPathPlanner:
    _ego_waypoint: Waypoint
    _goal_waypoint: Waypoint
    _local_planner: AStarPlanner
    _converter: FrameSegmentConverter
    _gps: CarlaGps
    _should_plan: bool

    def __init__(self, controller: CarlaSimulatorController) -> None:
        self._local_planner = AStarPlanner()
        self._ego_waypoint = Waypoint(200, 137)
        self._goal_waypoint = None
        self._converter = FrameSegmentConverter(400, 300)
        self._gps = CarlaGps(controller)
        self._simulation_controller = controller
        self._should_plan = True


    def _transmit_planned_path(self, bev, path: List[Waypoint]):

        print(f'shape: {bev.shape[1]} x {bev.shape[0]}')
        frame = self._converter.convert_clone_frame(
            bev, bev.shape[1], bev.shape[0])

        if not (path is None):
            for w in path:
                frame[w.z][w.x][0] = 255
                frame[w.z][w.x][1] = 255
                frame[w.z][w.x][2] = 255

        self._last_planned_frame = frame

    def _set_path(self, bev, path) -> any:

        res = self._converter.convert_clone_frame(
            bev, bev.shape[1], bev.shape[0])

        for w in path:
            res[w.z, w.x, :] = 255

        return res

    def move_car(self, curr_pos: Waypoint, new_goal: Waypoint):
        vehicle = self._simulation_controller.get_vehicle()
        if curr_pos.x != new_goal.x:
            steering_angle = math.atan(
                (new_goal.z - curr_pos.z)/(new_goal.x - curr_pos.x))
            vehicle.steer(steering_angle)
        if new_goal.z > curr_pos.z:
            vehicle.forward()
        else:
            vehicle.backward()

    def on_new_frame(self, bev) -> any:
        if not self._should_plan:
            return
        
        print(f"planning on frame: {bev.shape}")
        result = self._local_planner.plan(
            bev, 350000, self._ego_waypoint, self._goal_waypoint)

        if result.valid:
            self.move_car(result.start, result.goal)
        else:
            print("invalid path")
            cv2.imwrite('aaa.png', bev)

        print(self._gps.read())
        print(result.goal)
        print(result.start)

        if result.valid:
            print('valid path')
            return self._set_path(bev, result.path)
        elif result.timeout:
            print('A* timeout execution')
        # else:
            # print ('A* didnt find a path')

        return bev
    
    def set_should_plan(self, val: bool):
        self._should_plan = val


simulation = CarlaSimulatorController()
planner = LocalPathPlanner(simulation)
simulation.run(planner.on_new_frame)

def on_autonomous_driving_state_change(state: bool):
    planner.set_should_plan(state)

manual_control = RemoteController(MqttClient('127.0.0.1', 1883), simulation.get_vehicle().get_actor(), on_autonomous_driving_state_change)

while True:
    time.sleep(1000)