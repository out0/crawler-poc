import sys
sys.path.append("..")
sys.path.append("../..")
import unittest, math
from mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from planner.vehicle_pose import VehiclePose
from planner.waypoint import Waypoint
import numpy as np
from planner.occupancy_grid import OccupancyGrid


class TestOccupancyGrid(unittest.TestCase):


    def compute_euclidian_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx * dx + dy * dy)

    def print_vector_feasb(self, frame: np.array) -> None:
         k = frame.shape[2]
        
         for i in range(frame.shape[0]):
            print("| ", end='')
            for j in range(frame.shape[1]):
                print(f" {int(frame[i,j,k-1])}", end='')
            print("],")


    def test_distance_feasibility(self):
        frame = np.array([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ])


        feasible_expect = np.array([
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
        ])

        goal = Waypoint(5, 0)

        og = OccupancyGrid(frame, 0)
        og.set_goal(goal)

        f = og.get_frame()

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                self.assertEqual(f[i, j, 2], feasible_expect[i, j])
                expected_dist = self.compute_euclidian_distance(j, i, goal.x, goal.z)
                self.assertEqual(f[i, j, 1], expected_dist, f"not valid for ({i},{j}): computed: {f[i, j, 1]}, expected: {expected_dist}")

    def test_distance_feasibility_minimal_dist(self):
        frame = np.array([
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [25, 0, 0], [25, 0, 0], [25, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0]]
        ])


        feasible_expect = np.array([
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        ])
        goal = Waypoint(5, 0)

        og = OccupancyGrid(frame, 3)
        og.set_goal(goal)

        f = og.get_frame()

        #self.print_vector_feasb(f)

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                self.assertEqual(f[i, j, 2], feasible_expect[i, j])
                expected_dist = self.compute_euclidian_distance(j, i, goal.x, goal.z)
                self.assertEqual(f[i, j, 1], expected_dist, f"not valid for ({i},{j}): computed: {f[i, j, 1]}, expected: {expected_dist}")

    def test_distance_feasibility_curved(self):
        frame = np.array([
            [[24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        feasible_expect = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],

            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        ])

        goal = Waypoint(5, 0)

        og = OccupancyGrid(frame, 0)
        og.set_goal(goal)

        f = og.get_frame()

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                self.assertEqual(f[i, j, 2], feasible_expect[i, j])
                expected_dist = self.compute_euclidian_distance(j, i, goal.x, goal.z)
                self.assertEqual(f[i, j, 1], expected_dist, f"not valid for ({i},{j}): computed: {f[i, j, 1]}, expected: {expected_dist}")

    def test_distance_feasibility_curved_minimal_dist(self):
        frame = np.array([
            [[24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [24, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])

        feasible_expect = np.array([
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        ])

        goal = Waypoint(5, 0)

        og = OccupancyGrid(frame, 1)
        og.set_goal(goal)

        f = og.get_frame()

        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                self.assertEqual(f[i, j, 2], feasible_expect[i, j])
                expected_dist = self.compute_euclidian_distance(j, i, goal.x, goal.z)
                self.assertEqual(f[i, j, 1], expected_dist, f"not valid for ({i},{j}): computed: {f[i, j, 1]}, expected: {expected_dist}")


if __name__ == "__main__":
    unittest.main()