from model.vehicle_pose import VehiclePose
from motion.reference_path import ReferencePath, RangeResult
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
 
        pos = last_pos
        lower_dist = 999999999
        best_pos = pos
        jumps = 0

        while pos < (len(self._path) - 1):
            self._ref_path.update_path(self._path[pos], self._path[pos + 1])

            result = self._ref_path.check_in_range(curr_pose, max_range_squared)

            if result.is_between_ref_points_in_path:
                if result.proportion_to_p2 <= 0.1:
                    #print(f"[path finder] pos: {last_pos} proportion_to_p2 <= 0.1: {result.proportion_to_p2}")
                    return pos + 1
                return pos

            else:
                if lower_dist > result.estimated_dist:
                    best_pos = pos
                lower_dist = min(lower_dist, result.estimated_dist)
                  
            pos += 1
            jumps += 1

            if jumps > 5:
                #print("[path finder] pos: {last_pos} too much jumps")
                break

        #print(f"[path finder] pos: {last_pos} not in range, the best option is {best_pos}")
        return best_pos


