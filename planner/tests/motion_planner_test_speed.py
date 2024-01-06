import sys
sys.path.append("..")
sys.path.append("../..")
import unittest, math
from planner.vehicle_pose import VehiclePose
from planner.stub_slam import SimpleSlam
from carlasim.carla_sim_controller import CarlaSimulatorController
from planner.motion.velocity_controller import VelocityController
import gi, time, math
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
Gst.init(None)

class TestMotionPlannerSpeed(unittest.TestCase):
    
    def test_constant_speed(self):
        simulation = CarlaSimulatorController()
        simulation.start('Town07', VehiclePose(0, 0, 0))
        slam = SimpleSlam(400, 300, 40, 70)

        c1 = VelocityController(simulation.get_vehicle(), 1.0)
        c1.set_speed(10)

        while True:
            print(simulation.get_vehicle().get_speed())
            time.sleep(0.5)


        # time.sleep(20)
        # c1.stop()

    
if __name__ == "__main__":
    unittest.main()

