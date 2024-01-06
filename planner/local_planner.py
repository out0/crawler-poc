from motion.longitudinal_controller import LongitudinalController
from motion.lateral_controller import LateralController
from model.slam import SLAM
from model.global_planner import GlobalPlanner
from .occupancy_grid import OccupancyGrid
from .astar import AStarPlanner

from carlasim.ego_car import EgoCar

LONGITUDINAL_CONTROLLER_SAMPLE_PERIOD_ms=150
LATERAL_CONTROLLER_SAMPLE_PERIOD_ms=50
LOOKAHEAD_DIST = 30.0

STATE_STOPPED = 0
STATE_PLANNING = 1
STATE_MOVING = 2

class LocalPlanner:
    _ego_car: EgoCar
    _acc: LongitudinalController
    _lateral_controller : LateralController
    _state: int


    def __init__(self, ego: EgoCar, slam: SLAM) -> None:
        self._ego_car = ego
        self._slam = slam
        
        self._acc = LongitudinalController(
            sampling_period_ms=LONGITUDINAL_CONTROLLER_SAMPLE_PERIOD_ms,
            odometer=lambda : self._ego_car.odometer.read(),
            power_actuator=lambda p: self._ego_car.set_power(p),
            brake_actuator=lambda p: self._ego_car.set_brake(p))

        self._lateral_controller = LateralController(
            period_ms=LATERAL_CONTROLLER_SAMPLE_PERIOD_ms,
            desired_look_ahead_dist=LOOKAHEAD_DIST,
            odometer=lambda : self._ego_car.odometer.read(),
            set_steering_angle=lambda p: self._ego_car.set_steering(p),
            slam_find_current_pose=lambda : self._slam.estimate_ego_pose()
        )

        self._acc.set_speed(0.0)
        self._lateral_controller.set_waypoint_list(None)

        self._acc.start()
        self._lateral_controller.start()

        self._state = STATE_STOPPED

    def start(self):
        pass

    def destroy(self):
        self._acc.destroy()
        self._lateral_controller.destroy()