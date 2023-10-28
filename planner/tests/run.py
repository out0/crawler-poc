import sys
sys.path.append("..")
sys.path.append("../..")
import unittest
from mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from planner.vehicle_pose import VehiclePose
from planner.waypoint import Waypoint
import math


class TestMapCoordinateConverter(unittest.TestCase):
    
    def test_diagonals_local_to_map(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)

        # (20, 20) is our 500x500 window center (250,250) 
        # in world coordinates
        self_location = VehiclePose(20, 20, math.radians(0))
        expected_var = 50

        expected_degree = 45
        world_pose = converter.convert_to_world_pose(self_location, Waypoint(500, 0))

        self.assertEqual(20 + expected_var, world_pose.x)
        self.assertEqual(20 + expected_var, world_pose.y)
        self.assertEqual(expected_degree, math.degrees(world_pose.heading))

        expected_degree = 135
        world_pose = converter.convert_to_world_pose(self_location, Waypoint(0, 0))

        self.assertEqual(20 - expected_var, world_pose.x)
        self.assertEqual(20 + expected_var, world_pose.y)
        self.assertEqual(expected_degree, math.degrees(world_pose.heading))

        expected_degree = -45
        world_pose = converter.convert_to_world_pose(self_location, Waypoint(500, 500))

        self.assertEqual(20 + expected_var, world_pose.x)
        self.assertEqual(20 - expected_var, world_pose.y)
        self.assertEqual(expected_degree, math.degrees(world_pose.heading))

        expected_degree = -135
        world_pose = converter.convert_to_world_pose(self_location, Waypoint(0, 500))

        self.assertEqual(20 - expected_var, world_pose.x)
        self.assertEqual(20 - expected_var, world_pose.y)
        self.assertEqual(expected_degree, math.degrees(world_pose.heading))

    def test_diagonals_map_to_local(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)

        # (20, 20) is our 500x500 window center (250,250) 
        # in world coordinates
        self_location = VehiclePose(20, 20, math.radians(0))

        p = converter.convert_to_waypoint(self_location, VehiclePose(70, 70, math.radians(45)))
        self.assertEqual(500, p.x)
        self.assertEqual(0, p.z)

        p = converter.convert_to_waypoint(self_location, VehiclePose(-30, 70, math.radians(45)))
        self.assertEqual(0, p.x)
        self.assertEqual(0, p.z)

        p = converter.convert_to_waypoint(self_location, VehiclePose(70, -30, math.radians(45)))
        self.assertEqual(500, p.x)
        self.assertEqual(500, p.z)

        p = converter.convert_to_waypoint(self_location, VehiclePose(-30, -30, math.radians(45)))
        self.assertEqual(0, p.x)
        self.assertEqual(500, p.z)

    def test_local_with_heading(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        
        self_location = VehiclePose(20, 20, math.radians(-45))
        pose = VehiclePose(30, 30, 45)

        p = converter.convert_to_waypoint(self_location, pose)
        print(f"{p}")


        pose2 = converter.convert_to_world_pose(self_location, p)

        self.assertAlmostEqual(pose2.x, pose.x, delta=0.5)
        self.assertAlmostEqual(pose2.y, pose.y, delta=0.5)

    def test_self_location(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        
        self_location = VehiclePose(20, 20, math.radians(-45))
        pose = VehiclePose(20, 20, 45)

        p = converter.convert_to_waypoint(self_location, pose)
        print(f"{p}")


        pose2 = converter.convert_to_world_pose(self_location, p)

        self.assertAlmostEqual(pose2.x, pose.x, delta=0.5)
        self.assertAlmostEqual(pose2.y, pose.y, delta=0.5)



    

if __name__ == "__main__":
    unittest.main()