from model.vehicle_pose import VehiclePose
from motion.reference_path import ReferencePath
from typing import List

class PathPositionFinder:
    _ref_path: ReferencePath
    _path: List[VehiclePose]

    def __init__(self, path: List[VehiclePose]) -> None:
        self.set_path(path)

    def set_path(self, path: List[VehiclePose]):
        if len(path) < 3:
            raise Exception("a reference path must have at least 3 points.")
        
        self._path = path
        self._ref_path = ReferencePath(path[0], path[1])
        

    def find_next_pos(self, last_pos: int, max_range_squared: float, curr_pose: VehiclePose) -> int:
 
        while last_pos < (len(self._path) - 1):
            self._ref_path.update_path(self._path[last_pos], self._path[last_pos + 1])

            in_range, percent_left_to_end = self._ref_path.check_in_range(curr_pose, max_range_squared)

            if in_range:
                if percent_left_to_end <= 0.1:
                    last_pos += 1
                
                print (f"in range for [{last_pos} - {last_pos+1}]")
                return last_pos
            
            last_pos += 1

        return -1


