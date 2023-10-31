import numpy as np
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
# DIR_LEFT = 3
# DIR_RIGHT = 4
# DIR_BOTTOM_LEFT = 5
# DIR_BOTTOM = 6
# DIR_BOTTOM_RIGHT = 7
DIR_BOTTOM_LEFT = 3
DIR_BOTTOM = 4
DIR_BOTTOM_RIGHT = 5


COMPUTE_DIRECTION_POS = [
    Waypoint(0, -1),
    Waypoint(-1, -1),
    Waypoint(1, -1),
    # Waypoint(-1, 0),
    # Waypoint(1, 0),
    Waypoint(0, 1),
    Waypoint(-1, 1),
    Waypoint(1, 1),
]

MOVING_COST = [
    1,  # up
    4,  # diag up
    4,  # diag up
    # 200,  # side
    # 200,  # side
    4,  # down!
    4,  # down!
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

    def classIsRoad(pclass: int) -> bool:
        return pclass == 1 or pclass == 24 or pclass == 25 or pclass == 26 or pclass == 27 or pclass == 14
    

    def _checkObstacle(self, frame: np.array, point: Waypoint) -> bool:
        if point.z < 0 or point.z >= frame.shape[0]:
            return True
        if point.x < 0 or point.x >= frame.shape[1]:
            return True
        if frame[point.z][point.x][2] == 0:
            return True
        return not AStarPlanner.classIsRoad(frame[point.z][point.x][0])


    def _searchValidXWaypoint(self,
                              bev_frame: np.array,
                              p: Waypoint) -> Union[Waypoint, None]:

        search_point = Waypoint(p.x - 1, p.z)

        while search_point.x >= 0:
            if not self._checkObstacle(bev_frame, search_point):
                return search_point
            search_point.x = search_point.x - 1

        search_point = Waypoint(p.x + 1, p.z)

        while search_point.x < bev_frame.shape[0]:
            if not self._checkObstacle(bev_frame, search_point):
                return search_point
            search_point.x = search_point.x + 1

        return None
    
    def _searchValidZWaypoint(self,
                              bev_frame: np.array,
                              p: Waypoint) -> Union[Waypoint, None]:

        search_point = Waypoint(p.x, p.z - 1)

        while search_point.z > 0:
            if not self._checkObstacle(bev_frame, search_point):
                return search_point
            search_point.z -= 1

        return None

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

    def _find_best_x(self, frame, z):
        best_lower_x = -1
        best_higher_x = -1
        best_size = 0
        
        x_init = -1
        x_end = 0
        len = frame.shape[1] - 1
        i = 0
        
        while i < len:
            if AStarPlanner.classIsRoad(frame[z, i, 0]):
                if x_init < 0:
                    x_init = i
                    x_end = i
                    i += 1

                    while i < len and AStarPlanner.classIsRoad(frame[z, i, 0]):
                        x_end = i
                        i += 1
                    
                    if x_end - x_init > best_size:
                        best_lower_x = x_init
                        best_higher_x = x_end
                        best_size = x_end - x_init
                    
                    x_init = -1
            i += 1

        return 0.5 * (best_higher_x + best_lower_x)

    def _find_best_goal(self, frame, z = 0):
        best_x = self._find_best_x(frame, z)

        if best_x < 0:
            return self._find_best_goal(frame, z + 1)
        
        return Waypoint(best_x, z)

    def _timeout(self, start_time: int, max_exec_time_ms: int):
        return 1000*(time.time() - start_time) > max_exec_time_ms

    def plan(self,
             bev_frame: np.array,
             max_exec_time_ms: int,
             start: Waypoint,
             goal: Waypoint = None) -> PlannerResult:

        result = PlannerResult()

        result.start = start
        result.goal = goal
        
        # if self._checkObstacle(bev_frame, start):
        #     result.start = self._searchValidZWaypoint(bev_frame, start)
        #     if self._checkObstacle(bev_frame, start):
        #         return result

        if result.goal is None:
            result.goal = self._find_best_goal(bev_frame)
        elif self._checkObstacle(bev_frame, goal):
            result.goal = self._searchValidXWaypoint(bev_frame, goal)
        
        if result.goal is None:
            result.invalid_goal = True
            return result

        start_time = time.time()

        plan_grid = NpPlanGrid(bev_frame.shape[1], bev_frame.shape[0])
        
        plan_grid.set_costs(start, [0, 0, 0])
        #plan_grid.set_parent_by_coord(start, [-1, -1])

        open_list = PriorityQueue()
        open_list.put(QueuedPoint(start, 0))

        found_goal = None

        while found_goal is None and not open_list.empty():
            curr_point = open_list.get(block=False)

            if self._timeout(start_time, max_exec_time_ms):
                result.timeout = True
                return result

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

                # if goal.z == next_point.z:
                #     # any point in goal zone is valid for me!
                #     found_goal = next_point
                #     plan_grid.set_parent(next_point, curr_point)
                #     continue

                deviation_from_goal = abs(next_point.x - result.goal.x)
                if next_point.x - result.goal.x > 0:
                    inclination = (next_point.z - result.goal.z)/(next_point.x - result.goal.x)
                else:
                    inclination = 0

                if result.goal.z == next_point.z and deviation_from_goal < 20:
                    # any point in goal zone is valid for me!
                    found_goal = next_point
                    plan_grid.set_parent(next_point, curr_point)
                    continue


                g = curr_costs[G_CHEAPEST_COST_TO_PATH] + MOVING_COST[dir] + inclination
                if deviation_from_goal > 20:
                    g += 1000
                h = self._compute_euclidian_distance(next_point, result.goal)
                f = g + h

                next_costs = plan_grid.get_costs(next_point)

                if next_costs[F_CURRENT_BEST_GUESS] > f:
                    plan_grid.set_costs(next_point, [f, g, h])
                    plan_grid.set_parent(next_point, curr_point)
                    open_list.put(QueuedPoint(next_point, f))

        if found_goal is None:
            return result

        path: List[Waypoint] = []

        path.append(found_goal)
        
        p = found_goal        
        parent = plan_grid.get_parent(p)
        while parent.x >= 0:
            path.append(parent)
            p = parent
            parent = plan_grid.get_parent(p)

        path.append(start)
        path.reverse()

        result.path = path
        result.valid = True
        return result
