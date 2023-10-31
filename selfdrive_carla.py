#
# Self-drive prof of concept using the Carla simulator
#
import sys
from typing import List
import signal
from carlasim.carla_sim_controller import CarlaSimulatorController
from gi.repository import Gst, GLib
from typing import List
from carlasim.gps import CarlaGps
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
from planner.mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
import numpy as np
import cv2, os

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
Gst.init(None)

class SimulationConfig:
    file: str
    dist: float
    auto: bool
    execute_mission: bool
    town: str
    is_running: bool

    def __init__(self) -> None:
        self.file = None
        self.dist = None
        self.auto = False
        self.execute_mission = True
        self.town = 'Town07'
        self.is_running = True

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
    


def compute_euclidian_dist(l1, l2) -> float:
    dx = (l1.x - l2.x)
    dy = (l1.y - l2.y)
    return math.sqrt (dx*dx + dy*dy)

class MissionBuilder:

    def __build_mission_auto_capture_handler(car: EgoCar, conf: SimulationConfig) -> None:
        f = open(conf.file, 'w')
        last_location = car.get_location()
        f.write(f"{conf.town}\n")
        f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
        f.close()
        
        while sim_config.is_running:
            l = car.get_location()
            if compute_euclidian_dist(last_location, l) > conf.dist:
                last_location = l
                print (f"new goal point: ({last_location.x},{last_location.y})")
                f = open(conf.file, '+a')
                f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
                f.close()
            time.sleep(0.01)

    def build_mission_auto(conf: SimulationConfig) -> None:
        print (f"building simulation waypoint auto_capture for {conf.town} for every move of {conf.dist}")
        client = CarlaClient(conf.town)
        print("building the ego car")
        car = EgoCar(client)\
            .autopilot()\
            .build()
        car.set_pose(0,0,0)
        print("waiting 5s")
        time.sleep(5)
        print(f"start capturing to {conf.file}. Press <enter> to terminate")

        mission_auto_thr: threading.Thread
        mission_auto_thr = threading.Thread(None, MissionBuilder.__build_mission_auto_capture_handler, "build_mission_auto_capture", [car, conf])
        mission_auto_thr.start()

        input()
        print("terminating...")
        sim_config.is_running = False
        mission_auto_thr.join()
        car.destroy()

    def __on_autonomous_driving_state_change(val: bool):
        pass

    def build_mission_manual(conf: SimulationConfig) -> None:
        print (f"building simulation waypoint manual capture for {conf.town}")

        client = CarlaClient(conf.town)
        print("building the ego car")
        car = EgoCar(client)\
            .build()

        car.set_pose(0,0,0)
        print("waiting 5s")
        time.sleep(5)
        print(f"start capturing to {conf.file}.")
        print(f"Press enter for point capture")
        print(f"Press x + enter for exit")

        last_location = car.get_location()
        f = open(conf.file, 'w')
        print (f"capturing: {last_location}")
        f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
        f.close()

        manual_control = RemoteController(MqttClient('127.0.0.1', 1883), car.get_actor(), MissionBuilder.__on_autonomous_driving_state_change)

        input_str = ""
        while (input_str != "x"):
            input_str = input()
            if input_str == "x":
                continue
        
            l = car.get_location()
            if l.x == last_location.x and l.y == last_location.y:
                print ("nothing changed, the new point was ignored")
                continue

            last_location = l
            print (f"capturing: {last_location}")
            f = open(conf.file, '+a')
            f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
            f.close()

class TeleportMotionPlanner:
    _car: EgoCar
    _map_converter: MapCoordinateConverterCarla
    
    def __init__(self, car: EgoCar, og_width: int, og_height: int) -> None:
        self._car = car
        self._map_converter = MapCoordinateConverterCarla(36, 30, og_width, og_height)

    def move_to(self, location: VehiclePose, goal: Waypoint) -> None:
        pose = self._map_converter.convert_to_world_pose(location, goal)
        self._car.set_pose(pose.x, pose.y, pose.heading)


STATE_INITIALIZE = 1
STATE_PLANNING = 2
STATE_MOVING = 3
STATE_STOP = 4

class LocalPlanner:
    _og: OccupancyGrid
    _global_planner: SimpleGlobalPlanner
    _local_path_planner: AStarPlanner
    _bev_start_waypoint: Waypoint
    _slam: SimpleSlam
    _car: EgoCar
    _motion_planner: TeleportMotionPlanner
    _plan_thr: threading.Thread
    _plan_thr_running: bool
    _plan_state: int
    _vision_module: CarlaSimulatorController
    _planned_local_path: PlannerResult
    _frame_count: int

    def __init__(self, global_planner: SimpleGlobalPlanner, car: EgoCar, vision_module: CarlaSimulatorController) -> None:
        self._global_planner = global_planner
        self._bev_start_waypoint = None
        self._og = OccupancyGrid(30, 50)
        self._local_path_planner = AStarPlanner()
        self._car = car
        self._motion_planner = None
        self._plan_thr = None
        self._plan_thr_running = False
        self._plan_state  = STATE_INITIALIZE
        self._vision_module = vision_module
        self._planned_local_path = None
        self._frame_count = 0
    
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
                self._plan_run_state_planning()
            elif self._plan_state == STATE_MOVING:
                self._plan_run_state_moving()

    def _plan_run_state_initialize(self):
        frame = self._vision_module.get_frame()
        self._bev_start_waypoint = Waypoint(int(frame.shape[1]/2), int(frame.shape[0]/2))
        self._motion_planner = TeleportMotionPlanner(self._car, frame.shape[1], frame.shape[0])
        self._slam = SimpleSlam(frame.shape[1], frame.shape[0], 30, 50)
        self._plan_state = STATE_PLANNING
    
    def _plan_run_state_planning(self):
        frame = self._vision_module.get_frame()
        location = self._slam.get_current_pose(self._car)
        self._planned_local_path = self.plan(location, frame)

        if self._planned_local_path.valid:
            print ("valid path!")
            self._plan_state = STATE_MOVING
            ##
        else:
            print ("invalid path!")
            p = self._global_planner.get_next_waypoint(location, force_next=True)
            if p is None:
                self._plan_state = STATE_STOP

        self.plot(frame, self._bev_start_waypoint, self._planned_local_path.goal, self._planned_local_path.path)

        k = 1

    def _plan_run_state_moving(self):
        location = self._slam.get_current_pose(self._car)
        self._motion_planner.move_to(location, self._planned_local_path.goal)
        time.sleep(0.5)
        self._plan_state = STATE_PLANNING
        time.sleep(0.5)

    def plot (self, frame: np.ndarray, start: Waypoint, goal: Waypoint, path: List[Waypoint]):
        frame = self._og.get_color_frame(frame)
        frame[start.z, start.x, :] = [0, 255, 0]
        frame[goal.z, goal.x, :] = [0, 0, 255]

        if not path is None:
            for p in path:
                frame[p.z, p.x, :] = [255, 255, 255]
        
        cv2.imwrite(f"results/planning/plan_outp_{self._frame_count}.png", frame)
        self._frame_count += 1


    def plan(self, location: VehiclePose, frame: np.ndarray) -> PlannerResult:
        next_goal = self._global_planner.get_next_waypoint(location)
        local_goal, frame = self._og.find_best_local_goal_waypoint(frame, location, next_goal)
        #self.plot(frame, self._bev_start_waypoint, local_goal, None)
        return self._local_path_planner.plan(frame, 1000000, self._bev_start_waypoint, local_goal)
        

def execute_mission(conf: SimulationConfig) -> None:
    print ("starting simulation")
    global_planner = SimpleGlobalPlanner()
    global_planner.read_mission(conf.file)
    start_point = global_planner.get_next_waypoint(None)
    simulation.start(conf.town, start_point)
    
    print ("simulation started")
    car = simulation.get_vehicle()
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
    # if argc < 2:
    #     UserCmd._help(argv)
    #     exit(1)

    # config = UserCmd.proc_input(argc, argv)

    # if config.execute_mission:
    #     execute_mission(config)
    # elif config.auto:
    #     build_mission_auto(config)
    # else:
    #     build_mission_manual(config)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    main(len(sys.argv), sys.argv)
    