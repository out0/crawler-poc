#
#  This global planner reads and writes location points from/to a file
#
#  those location points can are fed to the local planner to perform a mission
#

from .waypoint import Waypoint
from io import TextIOWrapper
import os
from .vehicle_pose import VehiclePose

class SimpleGlobalPlanner:
    _file: str
    _pos: int
    _opened_file: TextIOWrapper

    def __init__(self, file: str) -> None:
        self._file = file
        self._pos = 1
        self._opened_file = None
        
    def check_exists(self) -> bool:
        return os.path.exists(self._file)
    
    def save_waypoint(self, p: Waypoint) -> None:
        if self._opened_file is not None:
            self._opened_file.close()
            self._opened_file = None

        f = open(file=self._file, mode='+a')
        f.write(f"{p.x}|{p.z}")
        f.close()

    def get_next_waypoint(self) -> VehiclePose:
        line: str = None
        if self._opened_file is None:
            self._opened_file = open(file=self._file, mode='+r')
            i = self._pos
            while i > 0:
                self._opened_file.readline()
                i -= 1
            line = self._opened_file.readline()
        else:
            self._pos += 1
            line = self._opened_file.readline()
        
        if line is None:
            return None
        
        vals = line.split('|')
        return VehiclePose(float(vals[0]),float(vals[1]),float(vals[2]))
        