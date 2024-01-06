import sys
sys.path.append("..")
sys.path.append("../..")
import unittest, math
from planner.vehicle_pose import VehiclePose
from model.waypoint import Waypoint
from planner.goal_point_discover import GoalPointDiscover
import numpy as np
from planner.occupancy_grid import OccupancyGrid
import cv2
from planner.astar import AStarPlanner
from typing import List

class TestGoalPointDiscover(unittest.TestCase):

    def read_frame(self, pos: int) -> np.array:
        return cv2.imread(f"test_data/planning_frame_{pos}.png")

    def write_waypoint_frame(self, frame: np.array, pos: int, point: Waypoint) -> np.array:
        frame[point.z, point.x, :] = [255, 255, 255]
        return cv2.imwrite(f"test_data/planning_frame_{pos}_result.png", frame)

    def convert_to_pose(self, line: str) -> VehiclePose:
        cols = line.split('|')
        return VehiclePose(float(cols[0]), float(cols[1]), float(cols[2]))

    def get_location_list(self) -> List[VehiclePose]:
        f = open("test_data/sim1.dat")
        lines = f.readlines()
        i = 1
        res = []
        while i < len(lines):
            pose = self.convert_to_pose(lines[i])
            res.append(pose)
            i += 1
        f.close()
        return res

    def exec_test_frame(self, minimal_distance: int, location_list: List[VehiclePose], pos: int) -> None:        
        planner = AStarPlanner()
        frame = self.read_frame(pos)
        start = Waypoint(int(frame.shape[1]/2), int(frame.shape[0]/2))
        location = location_list[pos - 1]
        next_global_point = location_list[pos]
        
        goal_discover = GoalPointDiscover(36, 30, minimal_distance)
        goal_discover.initialize(frame.shape)
        og = OccupancyGrid(frame, minimal_distance)
        local_goal_point = goal_discover.find_goal_waypoint(og, location, next_global_point)

        f_color = og.get_color_frame()            
        if local_goal_point is None:
            print(f"error processing frame {pos}")
            return
        planner_result = planner.plan(og.get_frame(), 99999999, start, local_goal_point)
        if planner_result.valid:
            print("valid path")
            for p in planner_result.path:
                f_color[p.z, p.x, :] = [255, 255, 255]
        else:
            print("invalid path")
        
        self.write_waypoint_frame(f_color, pos, local_goal_point)

    def test_frame_28(self):
        minimal_distance = 20
        location_list = self.get_location_list()
        self.exec_test_frame(minimal_distance, location_list, 28)

    # def test_all_frames(self):
    #     minimal_distance = 20
    #     location_list = self.get_location_list()

    #     for i in range(1, len(location_list)):
    #         self.exec_test_frame(minimal_distance, location_list, i)

if __name__ == "__main__":
    unittest.main()


