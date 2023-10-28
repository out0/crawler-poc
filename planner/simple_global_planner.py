#
#  This global planner reads and writes location points from/to a file
#
#  those location points can are fed to the local planner to perform a mission
#

from typing import List
import os, math
from .vehicle_pose import VehiclePose
from .waypoint import Waypoint


class SimpleMissionSaver:
    _file: str

    def _file_clear(self) -> None:
        f = open(file=self._file, mode='w')
        f.close()

    def __init__(self, file: str) -> None:
        self._file = file
        self._file_clear()
    
    def save_waypoint(self, p: Waypoint) -> None:
        f = open(file=self._file, mode='+a')
        f.write(f"{p.x}|{p.z}")
        f.close()

class SimpleGlobalPlanner:
    _goal_poses: List[VehiclePose]
    _pos: int

    def __init__(self) -> None:
        self._pos = -1
        self._goal_poses = []
  
    def _euclidian_dist(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return math.sqrt(dx*dx + dy*dy)

    def read_mission(self, file: str) -> bool:
        if not os.path.exists(file):
            print (f"mission file {file} not found")
            return False
        
        f = open(file=file, mode='+r')
        lines = f.readlines()
        
        skip = True
        for line in lines:
            if skip:
                skip = False
                continue
            vals = line.split('|')
            pose = VehiclePose(float(vals[0]),float(vals[1]),float(vals[2]))
            self._goal_poses.append(pose)

        self._pos = -1

        return True

    def get_next_waypoint(self, location: VehiclePose) -> VehiclePose:
        if self._pos < 0:
            self._pos = 0
            return self._goal_poses[self._pos]
        
        dist = self._euclidian_dist(location, self._goal_poses[self._pos])
        if dist < 20:
            self._pos += 1
        
        return self._goal_poses[self._pos]