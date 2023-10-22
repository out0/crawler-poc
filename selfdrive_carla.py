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
from planner.simple_global_planner import SimpleGlobalPlanner

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
Gst.init(None)

class SimulationConfig:
    file: str
    dist: float
    auto: bool
    execute_mission: bool
    town: str

    def __init__(self, file: str) -> None:
        self.file = file
        self.dist = None
        self.auto = False
        self.execute_mission = True
        self.town = 'Town07'

class SimulationKillSwitch:
    is_running: bool

    def __init__(self) -> None:
        self.is_running = True


class UserCmd:
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
    
    def proc_input(argc: int, argv: List[str]) -> SimulationConfig:
        if argc < 2:
            UserCmd._help(argv)
            return None
        
        conf = SimulationConfig(argv[1])

        i = 2
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
sim_ks = SimulationKillSwitch()


def handler(signum, frame):
    if simulation is not None:
        print ("\nterminating simulation")
        simulation.terminate()
        print ("the simulation is terminated")
    
    sim_ks.is_running = False
    

def on_new_frame(f: any) -> None:
    pass

def execute_mission(conf: SimulationConfig) -> None:
    print ("starting simulation")
    global_planner = SimpleGlobalPlanner(conf.file)
    simulation.start(conf.town, global_planner.get_next_waypoint())
    simulation.run(on_new_frame)
    print ("simulation started")

def compute_euclidian_dist(l1, l2) -> float:
    dx = (l1.x - l2.x)
    dy = (l1.y - l2.y)
    return math.sqrt (dx*dx + dy*dy)

def build_mission_auto_capture_handler(car: EgoCar, conf: SimulationConfig) -> None:
    f = open(conf.file, 'w')
    last_location = car.get_location()
    f.write(f"{conf.town}\n")
    f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
    f.close()
    
    while sim_ks.is_running:
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
    mission_auto_thr = threading.Thread(None, build_mission_auto_capture_handler, "build_mission_auto_capture", [car, conf])
    mission_auto_thr.start()

    input()
    print("terminating...")
    sim_ks.is_running = False
    mission_auto_thr.join()
    car.destroy()



def build_mission_manual(conf: SimulationConfig) -> None:
    print (f"building simulation waypoint manual capture for {conf.town}")
    pass




def main(argc: int, argv: List[str]):    
    if argc < 2:
        UserCmd._help(argv)
        exit(1)

    config = UserCmd.proc_input(argc, argv)

    if config.execute_mission:
        execute_mission(config)
    elif config.auto:
        build_mission_auto(config)
    else:
        build_mission_manual(config)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    main(len(sys.argv), sys.argv)
    