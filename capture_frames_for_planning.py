#
# Self-drive prof of concept using the Carla simulator
#
import sys
from typing import List
from carlasim.carla_sim_controller import CarlaSimulatorController
from typing import List
import gi, time
from planner.vehicle_pose import VehiclePose
import cv2, os

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)


def convert_to_waypoint(line: str) -> VehiclePose:
    cols = line.split('|')
    return VehiclePose(float(cols[0]), float(cols[1]), float(cols[2]))

def capture_frames(simulator_controller: CarlaSimulatorController, lines: List[str]):
    pos: int = 1

    simulator_controller.start('Town07', convert_to_waypoint(lines[pos]))
    print("waiting 5s")
    time.sleep(5)
    car = simulator_controller.get_vehicle()

    while pos < len(lines):
        point = convert_to_waypoint(lines[pos])
        print(f"moving to {point}")
        car.set_pose(point.x, point.y, point.heading)
        time.sleep(2)
        frame = simulator_controller.get_frame()
        print(f"writing frame {pos}")
        cv2.imwrite(f"results/dataset/planning_frame_{pos}.png", frame)
        time.sleep(1)
        pos += 1


def main(argc: int, argv: List[str]):
    
    # if argc <= 1 or not os.path.exists(argv[1]):
    #     print (f"use {argv[0]} <plan file>")
    #     return
    
    #file = argv[1]
    file = "sim1.dat"
    
    f = open(file, mode="r")
    lines = f.readlines()

    simulator_controller: CarlaSimulatorController = CarlaSimulatorController()
    capture_frames(simulator_controller, lines)

if __name__ == '__main__':
    main(len(sys.argv), sys.argv)
    