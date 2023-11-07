import time
from carlasim.carla_client import CarlaClient
from carlasim.vehicle_hal import EgoCar
import threading
from carlasim.mqtt_client import MqttClient
from carlasim.remote_controller import RemoteController
import math

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

class MissionBuilder:


    def __compute_euclidian_dist(l1, l2) -> float:
        dx = (l1.x - l2.x)
        dy = (l1.y - l2.y)
        return math.sqrt (dx*dx + dy*dy)

    def __build_mission_auto_capture_handler(car: EgoCar, conf: SimulationConfig) -> None:
        f = open(conf.file, 'w')
        last_location = car.get_location()
        f.write(f"{conf.town}\n")
        f.write(f"{last_location.x}|{last_location.y}|{car.get_heading()}\n")
        f.close()
        
        while conf.is_running:
            l = car.get_location()
            if MissionBuilder.__compute_euclidian_dist(last_location, l) > conf.dist:
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
        conf.is_running = False
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