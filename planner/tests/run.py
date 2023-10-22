import sys
sys.path.append("..")
sys.path.append("../..")
import unittest
from mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from planner.vehicle_pose import VehiclePose
from planner.waypoint import Waypoint

class TestMapCoordinateConverter(unittest.TestCase):
    
    def test_convert_self_location_world_coordinates(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)

        # (20, 20) is our 500x500 window center (250,250) 
        # in world coordinates
        self_location = VehiclePose(20, 20, 0)

        target = Waypoint(250, 250)
        
        world_pose = converter.convert_to_world_pose(self_location, target, heading=0)

        self.assertEqual(self_location.x, world_pose.x)
        self.assertEqual(self_location.y, world_pose.y)

    def test_convert_self_location_local_coordinates(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)

        # (20, 20) is our 500x500 window center (250,250) 
        # in world coordinates
        self_location = VehiclePose(20, 20, 0)
        
        local_center = Waypoint(250, 250)       
        local_waypoint = converter.convert_to_waypoint(self_location, self_location)

        self.assertEqual(local_waypoint.x, local_center.x)
        self.assertEqual(local_waypoint.z, local_center.z)

    def test_convert_from_and_back_location(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)

        # (20, 20) is our 500x500 window center (250,250) 
        # in world coordinates
        self_location = VehiclePose(20, 20, 0)


        for i in range (0, 499):
            for j in range (0, 499):
                local = Waypoint(i, j)
                world_pose = converter.convert_to_world_pose(self_location, local, heading=0)
                local_reconverted = converter.convert_to_waypoint(self_location, world_pose)

                self.assertAlmostEqual(local.x, local_reconverted.x, delta=1)
                self.assertAlmostEqual(local.z, local_reconverted.z, delta=1)
    

    

if __name__ == "__main__":
    unittest.main()