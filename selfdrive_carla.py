#
# Self-drive prof of concept using the Carla simulator
#
import sys
from typing import List
import signal
from carlasim.carla_sim_controller import CarlaSimulatorController
from gi.repository import Gst, GLib
from typing import List
import gi, time, math
from carlasim.carla_client import CarlaClient
from carlasim.vehicle_hal import EgoCar
import threading
from planner.simple_global_planner import SimpleGlobalPlanner, SimpleMissionSaver
from carlasim.mqtt_client import MqttClient
from carlasim.remote_controller import RemoteController
from planner.astar import AStarPlanner, PlannerResult
from planner.waypoint import Waypoint
from planner.occupancy_grid import OccupancyGrid
from planner.vehicle_pose import VehiclePose
from planner.simple_slam import SimpleSlam
from planner.motion_planner import MotionPlanner
from misc.mission_builder import SimulationConfig, MissionBuilder

from planner.goal_point_discover import GoalPointDiscover
import numpy as np
import cv2, os

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
Gst.init(None)

class UserCmd:

    def __init__(self) -> None:
        pass

    def _help(argv: List[str]):
        print (f"use {argv[0]} <config file>\n\n")
        print (f"when no option is given, {argv[0]} will run the simulation described in <config file>\n")
        print ("-a [Town] [dist] \t- runs the simulation with auto pilot ON, collecting goal points whenever the ego-car moves 'dist'")
        print ("-m [Town]\t\t- runs the simulation with auto pilot OFF and opens a prompt for goal point collection\n\n")
    
    def _check_next_argument(i: int, argc: int, error_msg: str) -> bool:
        if i == argc - 1:
            print (error_msg)
            return False
        return True
    
    def proc_input(conf: SimulationConfig, argc: int, argv: List[str]) -> None:
        if argc < 2:
            UserCmd._help(argv)
            return None
        
        i = 1
        
        if argv[1][i] == '-':
            conf.file = 'sim_plan.dat'
        else:
            conf.file = argv[1]
            i += 1
        
        while i < argc:
            if argv[i] == '-a':
                conf.dist = 2
                conf.auto = True
                conf.execute_mission = False

                if argc > i + 1 and argv[i+1][0] != '-':
                    i += 1
                    conf.town = argv[i]
                if argc > i + 1 and argv[i+1][0] != '-':
                    i += 1
                    conf.dist = int(argv[i])

            elif argv[i] == '-m':
                conf.dist = None
                conf.auto = False
                conf.execute_mission = False
                
                if argc > i + 1 and argv[i+1][0] != '-':
                    i += 1
                    conf.town = argv[i]
            i += 1
        
        return conf    

simulation = CarlaSimulatorController()
sim_config = SimulationConfig()

def handler(signum, frame):
    if simulation is not None:
        print ("\nterminating simulation")
        simulation.terminate()
        print ("the simulation is terminated")
    
    sim_config.is_running = False
    


STATE_INITIALIZE = 1
STATE_PLANNING = 2
STATE_MOVING = 3
STATE_STOP = 4
STATE_SLOW_PLAN = 5

CAR_WIDTH = 40
CAR_HEIGHT = 70
MINIMAL_DISTANCE = 10

class LocalPlanner:
    _global_planner: SimpleGlobalPlanner
    _local_path_planner: AStarPlanner
    _bev_start_waypoint: Waypoint
    _slam: SimpleSlam
    _car: EgoCar
    _motion_planner: MotionPlanner
    _plan_thr: threading.Thread
    _plan_thr_running: bool
    _plan_state: int
    _vision_module: CarlaSimulatorController
    _planned_local_path: PlannerResult
    _frame_count: int
    _goal_point_discover: GoalPointDiscover

    def __init__(self, global_planner: SimpleGlobalPlanner, car: EgoCar, vision_module: CarlaSimulatorController) -> None:
        self._global_planner = global_planner
        self._bev_start_waypoint = None
        self._local_path_planner = AStarPlanner()
        self._car = car
        self._motion_planner = None
        self._plan_thr = None
        self._plan_thr_running = False
        self._plan_state  = STATE_INITIALIZE
        self._vision_module = vision_module
        self._planned_local_path = None
        self._frame_count = 0
        self._goal_point_discover = GoalPointDiscover(CAR_WIDTH, CAR_HEIGHT, MINIMAL_DISTANCE)
    
    def start(self):
        _plan_thr = threading.Thread(None, self._plan_run)
        _plan_thr.start()

    def terminate(self):
        self._plan_thr_running = False
        if self._plan_thr != None:
            self._plan_thr.join()
        self._plan_thr = None

    def _plan_run(self):
        self._plan_thr_running = True
        
        while self._plan_thr_running:
            if self._plan_state == STATE_INITIALIZE:
                self._plan_run_state_initialize()
            elif self._plan_state == STATE_PLANNING:
                self._plan_run_state_planning(25000000)
            elif self._plan_state == STATE_MOVING:
                self._plan_state = self._plan_run_state_moving()
            elif self._plan_state == STATE_SLOW_PLAN:
                self._motion_planner.stop()
                self._plan_run_state_planning(100000000)

    def _plan_run_state_initialize(self):
        frame = self._vision_module.get_frame()
        self._bev_start_waypoint = Waypoint(int(frame.shape[1]/2), int(frame.shape[0]/2))
        self._slam = SimpleSlam(frame.shape[1], frame.shape[0], CAR_WIDTH, CAR_HEIGHT)
        self._plan_state = STATE_PLANNING
        self._goal_point_discover.initialize(frame.shape)
        self._motion_planner = MotionPlanner(self._car, self._goal_point_discover.get_map_coordinate_converter())
    
    def _plan_run_state_planning(self, timeout_ms: int):
        og = OccupancyGrid(self._vision_module.get_frame(), MINIMAL_DISTANCE)
        location = self._slam.get_current_pose(self._car)
        self._planned_local_path = self.plan(location, timeout_ms, og)

        if self._planned_local_path.valid:
            print ("valid path!")
            self._plan_state = STATE_MOVING
            self._motion_planner.move_on_path(location, self._planned_local_path.global_goal, self._planned_local_path.path)
            ##
        else:
            print ("invalid path!")
            self._plan_state = STATE_SLOW_PLAN
            # p = self._global_planner.get_next_waypoint(location, force_next=True)
            # if p is None:
            #     self._plan_state = STATE_STOP
        
        self.plot(og, self._bev_start_waypoint, self._planned_local_path.goal, self._planned_local_path.path)

        

    def _plan_run_state_moving(self):
        if self._motion_planner.is_moving():
            return STATE_MOVING
        return STATE_PLANNING

    def plot (self, og: OccupancyGrid, start: Waypoint, goal: Waypoint, path: List[Waypoint]):
        frame = og.get_color_frame()
        frame[start.z, start.x, :] = [0, 255, 0]

        if goal is not None:
            frame[goal.z, goal.x, :] = [255, 255, 255]

        if not path is None:
            for p in path:
                frame[p.z, p.x, :] = [255, 255, 255]
        
        cv2.imwrite(f"results/planning/plan_outp_{self._frame_count}.png", frame)
        self._frame_count += 1

    def plan(self, location: VehiclePose, timeout_ms: int, og: OccupancyGrid) -> PlannerResult:
        next_goal = self._global_planner.get_next_waypoint(location)
        print (f"planning to reach {next_goal}")
        next_local_goal = self._goal_point_discover.find_goal_waypoint(og, location, next_goal)
        og.set_goal(next_local_goal)
        result = self._local_path_planner.plan(og.get_frame(), timeout_ms, self._bev_start_waypoint, next_local_goal)
        result.global_goal = next_goal
        return result
        

def execute_mission(conf: SimulationConfig) -> None:
    print ("starting simulation")
    global_planner = SimpleGlobalPlanner()
    global_planner.read_mission(conf.file)
    start_point = global_planner.get_next_waypoint(None)
    simulation.start(conf.town, start_point)
    
    print ("simulation started")
    car = simulation.get_vehicle()
    print (f"teleporting to {start_point}")
    car.set_pose(start_point.x, start_point.y, start_point.heading)
    
    # p = start_point
    # while p is not None:
    #     location = car.get_location()
    #     p = global_planner.get_next_waypoint(location)
    #     car.set_pose(p.x, p.y, p.heading)
    #     time.sleep(2)
    # return

    local_planner = LocalPlanner(global_planner, car, simulation)
    print("waiting 5s")
    time.sleep(5)
    local_planner.start()



def main(argc: int, argv: List[str]):    
    config = SimulationConfig()
    config.file =  "sim1.dat"
    config.auto = False
    execute_mission(config)
    return

    # if argc < 2:
    #     UserCmd._help(argv)
    #     exit(1)

    # config = UserCmd.proc_input(config, argc, argv)

    # if config.execute_mission:
    #     execute_mission(config)
    # elif config.auto:
    #     MissionBuilder.build_mission_auto(config)
    # else:
    #     MissionBuilder.build_mission_manual(config)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    main(len(sys.argv), sys.argv)
    