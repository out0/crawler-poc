import numpy as np

from .vehicle_pose import VehiclePose
from .waypoint import Waypoint
from typing import List, Union
import sys
import math
import time
from queue import PriorityQueue
from io import StringIO

MAX_FLOAT = sys.float_info.max - 10

DIR_TOP = 0
DIR_TOP_LEFT = 1
DIR_TOP_RIGHT = 2
DIR_LEFT = 3
DIR_RIGHT = 4
DIR_BOTTOM_LEFT = 5
DIR_BOTTOM = 6
DIR_BOTTOM_RIGHT = 7


COMPUTE_DIRECTION_POS = [
    Waypoint(0, -1),
    Waypoint(-1, -1),
    Waypoint(1, -1),
    Waypoint(-1, 0),
    Waypoint(1, 0),
    Waypoint(0, 1),
    Waypoint(-1, 1),
    Waypoint(1, 1),
]

MOVING_COST = [
    1,  # up
    5,  # diag up
    5,  # diag up
    50,  # side
    50,  # side
    200,  # down!
    200,  # down!
    500  # down!
]


class QueuedPoint(Waypoint):
    cost: float

    def __init__(self, p: Waypoint, cost: float):
        super().__init__(p.x, p.z)
        self.cost = cost

    def __gt__(self, other):
        return self.cost > other.cost

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost


class NpPlanGrid:
    grid: np.array
    costs: np.array

    def __init__(self, width: int, height: int):
        self.grid = np.full((height, width, 3), -1)
        self.costs = np.full((height, width, 3), MAX_FLOAT)

    def set_closed(self, point: Waypoint) -> None:
        self.grid[point.z, point.x, 0] = 1

    def is_closed(self, point: Waypoint) -> bool:
        return self.grid[point.z, point.x, 0] == 1

    def set_parent(self, point: Waypoint, parent: Waypoint) -> None:
        self.grid[point.z, point.x, 1] = parent.x
        self.grid[point.z, point.x, 2] = parent.z

    def set_parent_by_coord(self, point: Waypoint, coord: List[int]) -> None:
        self.grid[point.z, point.x, 1] = coord[0]
        self.grid[point.z, point.x, 2] = coord[1]


    def get_costs(self, point: Waypoint) -> np.array:
        return self.costs[point.z, point.x]
    
    def set_costs(self, point: Waypoint, lst: List) -> None:
        self.costs[point.z, point.x] = lst

    def get_parent(self, point: Waypoint) -> Waypoint:
        return Waypoint(self.grid[point.z, point.x][1], self.grid[point.z, point.x][2])
    
F_CURRENT_BEST_GUESS = 0
G_CHEAPEST_COST_TO_PATH = 1
H_DISTANCE_TO_GOAL = 2

class PlannerResult:
    path: List[Waypoint]
    start: Waypoint
    goal: Waypoint
    invalid_goal: bool
    valid: bool
    timeout: bool
    global_goal: VehiclePose

    def __init__(self) -> None:
        self.valid = False
        self.timeout = False
        self.invalid_goal = False
        self.path = None
    
    def __str__(self) -> str:
        str = StringIO()
        str.write("{\n")
        str.write(f"    start: {(self.start.x, self.start.z)}\n")
        if self.invalid_goal:
            str.write(f"    goal: (INVALID)\n")
        else:
            str.write(f"    goal: {(self.goal.x, self.goal.z)}\n")
        str.write("    path: ")

        if self.valid:
            first = True
            for p in self.path:
                if not first:
                    str.write(" -> ")
                str.write(f"{(p.x, p.z)}")
                first = False
            str.write("\n")
        else:
            str.write("(INVALID)\n")
        
        str.write("}\n")

        if self.timeout:
            str.write(f"    timeout: {self.timeout}")

        return str.getvalue()


class AStarPlanner:

    def _checkObstacle(self, frame: np.array, point: Waypoint) -> bool:
        if point.z < 0 or point.z >= frame.shape[0]:
            return True
        if point.x < 0 or point.x >= frame.shape[1]:
            return True
        return frame[point.z][point.x][2] == 0
    
    def _add_points(self, p1: Waypoint, p2: Waypoint) -> Waypoint:
        return Waypoint(p1.x + p2.x, p1.z + p2.z)

    def _compute_free_surroundings(self,
                                   frame: np.array,
                                   point: Waypoint) -> List[bool]:
        res: List[bool] = []

        for _ in range(0, 8):
            res.append(False)

        res[DIR_TOP] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP])
        )
        res[DIR_TOP_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_LEFT])
        )
        res[DIR_TOP_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_TOP_RIGHT])
        )
        res[DIR_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_LEFT])
        )
        res[DIR_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_RIGHT])
        )                
        res[DIR_BOTTOM_LEFT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_LEFT])
        )
        res[DIR_BOTTOM] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM]))
        res[DIR_BOTTOM_RIGHT] = not self._checkObstacle(
            frame,
            self._add_points(point, COMPUTE_DIRECTION_POS[DIR_BOTTOM_RIGHT])
        )

        return res

    def _compute_euclidian_distance(self, p1: Waypoint, p2: Waypoint) -> float:
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.sqrt(dz * dz + dx * dx)

       
    def _timeout(self, start_time: int, max_exec_time_ms: int):
        return 1000*(time.time() - start_time) > max_exec_time_ms

    def plan(self,
             bev_frame: np.array,
             max_exec_time_ms: int,
             start: Waypoint,
             goal: Waypoint) -> PlannerResult:

        result = PlannerResult()

        result.start = start
        result.goal = goal
        
        if result.goal is None:
            result.invalid_goal = True
            return result

        start_time = time.time()
        plan_grid = NpPlanGrid(bev_frame.shape[1], bev_frame.shape[0])
        
        plan_grid.set_costs(start, [0, 0, 0])
        #plan_grid.set_parent_by_coord(start, [-1, -1])

        open_list = PriorityQueue()
        open_list.put(QueuedPoint(start, 0))

        best_possible = None
        best_distance_to_goal: float = MAX_FLOAT
        search = True

        while search and not open_list.empty():
            curr_point = open_list.get(block=False)

            if self._timeout(start_time, max_exec_time_ms):
                search = False
                continue

            plan_grid.set_closed(curr_point)

            free_surroundings = self._compute_free_surroundings(
                bev_frame, curr_point)

            f: float = MAX_FLOAT
            g: float = MAX_FLOAT
            h: float = MAX_FLOAT

            curr_costs = plan_grid.get_costs(curr_point)

            for dir in range(0, 6):
                if not free_surroundings[dir]:
                    continue

                next_point = self._add_points(
                    curr_point, COMPUTE_DIRECTION_POS[dir])
                
                if plan_grid.is_closed(next_point):
                    continue

                distance_to_goal = bev_frame[next_point.z, next_point.x, 1]

                if distance_to_goal < best_distance_to_goal:
                    best_distance_to_goal = distance_to_goal
                    best_possible = next_point
                
                if next_point.x == goal.x and next_point.z == goal.z:
                    best_possible = next_point
                    best_distance_to_goal = 0
                    search = False

                g = curr_costs[G_CHEAPEST_COST_TO_PATH] + MOVING_COST[dir] - bev_frame[next_point.z, next_point.x, 2]
                h = distance_to_goal
                f = g + h

                next_costs = plan_grid.get_costs(next_point)

                if next_costs[F_CURRENT_BEST_GUESS] > f:
                    plan_grid.set_costs(next_point, [f, g, h])
                    plan_grid.set_parent(next_point, curr_point)
                    open_list.put(QueuedPoint(next_point, f))

        if best_possible is None:
            result.invalid_goal = True
            result.valid = False
            return result

        path: List[Waypoint] = []

        path.append(best_possible)
        
        p = best_possible        
        parent = plan_grid.get_parent(p)
        while parent.x >= 0:
            path.append(parent)
            p = parent
            parent = plan_grid.get_parent(p)

        path.append(start)
        path.reverse()

        result.path = path
        result.valid = True
        result.goal = best_possible
        return result
