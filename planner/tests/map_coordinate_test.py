import sys
sys.path.append("..")
sys.path.append("../..")
import unittest, math
from mapping.map_coordinate_converter_carla import MapCoordinateConverterCarla
from planner.vehicle_pose import VehiclePose
from model.waypoint import Waypoint

class TestMapCoordinate(unittest.TestCase):

    BEV_DIRECTIONS = [
        Waypoint(0, 0),         # top-left
        Waypoint(250, 0),       # top
        Waypoint(500, 0),       # top-right
        Waypoint(0, 250),       # left
        Waypoint(250, 250),     # center
        Waypoint(500, 250),     # right
        Waypoint(0, 500),       # bottom-left
        Waypoint(250, 500),     # bottom
        Waypoint(500, 500),     # bottom-right
    ]

    def test_to_map_center_origin_no_heading(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(0, 0, 0)

        expected = [
            VehiclePose(50, -50, -45),    # top-left
            VehiclePose(50, 0, 0),      # top
            VehiclePose(50, 50, 45),     # top-right
            VehiclePose(0, -50, -90),     # left
            VehiclePose(0, 0, 0),       # center
            VehiclePose(0, 50, 90),      # right
            VehiclePose(-50, -50, -135),   # bottom-left
            VehiclePose(-50, 0, 180),     # bottom
            VehiclePose(-50, 50, 135),    # bottom-right
        ]

        i = 0
        for i in range (len(TestMapCoordinate.BEV_DIRECTIONS)):
            computed = converter.convert_to_world_pose(center, TestMapCoordinate.BEV_DIRECTIONS[i])
            self.assertAlmostEqual(expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].y, computed.y, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].heading, computed.heading, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")

    def test_from_map_center_origin_no_heading(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(0, 0, 0)

        values = [
            VehiclePose(50, -50, -45),    # top-left
            VehiclePose(50, 0, 0),      # top
            VehiclePose(50, 50, 45),     # top-right
            VehiclePose(0, -50, -90),     # left
            VehiclePose(0, 0, 0),       # center
            VehiclePose(0, 50, 90),      # right
            VehiclePose(-50, -50, -135),   # bottom-left
            VehiclePose(-50, 0, 180),     # bottom
            VehiclePose(-50, 50, 135),    # bottom-right
        ]

        i = 0
        expected = TestMapCoordinate.BEV_DIRECTIONS
        
        for i in range (len(TestMapCoordinate.BEV_DIRECTIONS)):
            computed = converter.convert_to_waypoint(center, values[i])           
            self.assertAlmostEqual(expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].z, computed.z, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
    
    
    def test_to_map_with_heading(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(0, 0, 45)

        p = math.sqrt(2)
        vector_size = 50 * p 
        r = math.radians(45)
        r2 = math.radians(135)
              

        expected = [
            VehiclePose(vector_size, 0, 0),                                             # top-left
            VehiclePose(50 * math.cos(r), 50 * math.sin(r), 45),                        # top
            VehiclePose(0, vector_size, 90),                                            # top-right
            VehiclePose(50 * math.cos(-r), 50 * math.sin(-r), -45),                     # left
            VehiclePose(0, 0, 0),                                                       # center
            VehiclePose(50 * math.cos(r2), 50 * math.sin(r2), 135),                     # right
            VehiclePose(0, -vector_size, -90),                                          # bottom-left
            VehiclePose(50 * math.cos(-r2), 50 * math.sin(-r2), -135),                  # bottom
            VehiclePose(-vector_size, 0, 180),                                          # bottom-right
        ]

        i = 0
        for i in range (len(TestMapCoordinate.BEV_DIRECTIONS)):
            computed = converter.convert_to_world_pose(center, TestMapCoordinate.BEV_DIRECTIONS[i])
            self.assertAlmostEqual(expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].y, computed.y, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].heading, computed.heading, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")

    def test_from_map_with_heading(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(0, 0, 45)

        p = math.sqrt(2)
        vector_size = 50 * p 
        r = math.radians(45)
        r2 = math.radians(135)
              
        expected = TestMapCoordinate.BEV_DIRECTIONS
        values = [
            VehiclePose(vector_size, 0, 0),                                             # top-left
            VehiclePose(50 * math.cos(r), 50 * math.sin(r), 45),                        # top
            VehiclePose(0, vector_size, 90),                                            # top-right
            VehiclePose(50 * math.cos(-r), 50 * math.sin(-r), -45),                     # left
            VehiclePose(0, 0, 0),                                                       # center
            VehiclePose(50 * math.cos(r2), 50 * math.sin(r2), 135),                     # right
            VehiclePose(0, -vector_size, -90),                                          # bottom-left
            VehiclePose(50 * math.cos(-r2), 50 * math.sin(-r2), -135),                  # bottom
            VehiclePose(-vector_size, 0, 180),                                          # bottom-right
        ]

        i = 0
        for i in range (len(values)):
            computed = converter.convert_to_waypoint(center, values[i])
            self.assertAlmostEqual(expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].z, computed.z, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")


    def test_to_map_with_heading_and_location(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(100, 100, 45)

        p = math.sqrt(2)
        vector_size = 50 * p 
        r = math.radians(45)
        r2 = math.radians(135)
              

        expected = [
            VehiclePose(vector_size, 0, 0),                                             # top-left
            VehiclePose(50 * math.cos(r), 50 * math.sin(r), 45),                        # top
            VehiclePose(0, vector_size, 90),                                            # top-right
            VehiclePose(50 * math.cos(-r), 50 * math.sin(-r), -45),                     # left
            VehiclePose(0, 0, 0),                                                       # center
            VehiclePose(50 * math.cos(r2), 50 * math.sin(r2), 135),                     # right
            VehiclePose(0, -vector_size, -90),                                          # bottom-left
            VehiclePose(50 * math.cos(-r2), 50 * math.sin(-r2), -135),                  # bottom
            VehiclePose(-vector_size, 0, 180),                                          # bottom-right
        ]

        i = 0
        for i in range (len(TestMapCoordinate.BEV_DIRECTIONS)):
            computed = converter.convert_to_world_pose(center, TestMapCoordinate.BEV_DIRECTIONS[i])
            self.assertAlmostEqual(100 + expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(100 + expected[i].y, computed.y, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].heading, computed.heading, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")

    def test_from_map_with_heading_and_location(self):
        converter = MapCoordinateConverterCarla(100, 100, 500, 500)
        center = VehiclePose(100, 100, 45)

        p = math.sqrt(2)
        vector_size = 50 * p 
        r = math.radians(45)
        r2 = math.radians(135)
              
        expected = TestMapCoordinate.BEV_DIRECTIONS
        values = [
            VehiclePose(vector_size, 0, 0),                                             # top-left
            VehiclePose(50 * math.cos(r), 50 * math.sin(r), 45),                        # top
            VehiclePose(0, vector_size, 90),                                            # top-right
            VehiclePose(50 * math.cos(-r), 50 * math.sin(-r), -45),                     # left
            VehiclePose(0, 0, 0),                                                       # center
            VehiclePose(50 * math.cos(r2), 50 * math.sin(r2), 135),                     # right
            VehiclePose(0, -vector_size, -90),                                          # bottom-left
            VehiclePose(50 * math.cos(-r2), 50 * math.sin(-r2), -135),                  # bottom
            VehiclePose(-vector_size, 0, 180),                                          # bottom-right
        ]

        i = 0
        for i in range (len(values)):
            computed = converter.convert_to_waypoint(center, VehiclePose(values[i].x + 100, values[i].y + 100, values[i].heading))
            self.assertAlmostEqual(expected[i].x, computed.x, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")
            self.assertAlmostEqual(expected[i].z, computed.z, delta=0.05, msg= f"wrong for {TestMapCoordinate.BEV_DIRECTIONS[i]}")

if __name__ == "__main__":
    unittest.main()