import numpy as np
from .waypoint import Waypoint
from typing import List, Union
import sys
import math
from queue import PriorityQueue

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
    500,  # down!
    500,  # down!
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


# class PlanCell:
#     g_cheapest_path_cost: float
#     f_current_best_guest_cost: float
#     h_dist_to_goal: float
#     parent_x: int
#     parent_z: int
#     is_closed: bool

#     def __init__(self):
#         self.g_cheapest_path_cost = MAX_FLOAT
#         self.f_current_best_guest_cost = MAX_FLOAT
#         self.h_dist_to_goal = MAX_FLOAT
#         self.parent_x = -1
#         self.parent_z = -1
#         self.is_closed = False

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
    
# class PlanGrid:
#     grid: List[List[PlanCell]]

#     def __init__(self, width: int, height: int):
#         self.grid = []
#         for _ in range(0, height):
#             l = []
#             for _ in range(0, width):
#                 l.append(PlanCell())
#             self.grid.append(l)

#     def set_closed(self, point: Waypoint) -> None:
#         self.grid[point.z][point.x].is_closed = True

#     def is_closed(self, point: Waypoint) -> bool:
#         return self.grid[point.z][point.x].is_closed

#     def set_parent(self, point: Waypoint, parent: Waypoint) -> None:
#         self.grid[point.z][point.x].parent_x = parent.x
#         self.grid[point.z][point.x].parent_z = parent.z

#     def get_cell(self, point: Waypoint) -> PlanCell:
#         return self.grid[point.z][point.x]


F_CURRENT_BEST_GUESS = 0
G_CHEAPEST_COST_TO_PATH = 1
H_DISTANCE_TO_GOAL = 2

class AStarPlanner:

    def _checkColorEquals(self, frame: np.array,
                          x: int,
                          z: int,
                          r: int,
                          g: int,
                          b: int) -> bool:
        return frame[z][x][0] == r \
            and frame[z][x][1] == g \
            and frame[z][x][2] == b

    def _checkObstacle(self, frame: np.array, point: Waypoint) -> bool:
        if point.x < 0 or point.x >= frame.shape[0]:
            return False
        if point.z < 0 or point.z >= frame.shape[1]:
            return False

        return self._checkColorEquals(frame, point.x, point.z, 0, 0, 0)

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
        # res[DIR_LEFT] = not self._checkObstacle(
        #     frame,
        #     self._add_points(point, COMPUTE_DIRECTION_POS[DIR_LEFT]))
        # res[DIR_RIGHT] = not self._checkObstacle(
        #     frame,
        #     self._add_points(point, COMPUTE_DIRECTION_POS[DIR_RIGHT])
        # )
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

    def plan(self,
             bev_frame: np.array,
             start: Waypoint,
             goal: Waypoint) -> Union[List[Waypoint], None]:

        if self._checkObstacle(bev_frame, start):
            return None

        if self._checkObstacle(bev_frame, goal):
            goal = self._searchValidXWaypoint(bev_frame, goal)
            if goal is None:
                return None

        plan_grid = NpPlanGrid(bev_frame.shape[1], bev_frame.shape[0])
        

        plan_grid.set_costs(start, [0, 0, 0])
        #plan_grid.set_parent_by_coord(start, [-1, -1])

        open_list = PriorityQueue()
        open_list.put(QueuedPoint(start, 0))

        found_goal = None

        while found_goal is None and not open_list.empty():
            curr_point = open_list.get(block=False)

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

                if goal.z == next_point.z:
                    # any point in goal zone is valid for me!
                    found_goal = next_point
                    plan_grid.set_parent(next_point, curr_point)
                    continue

                g = curr_costs[G_CHEAPEST_COST_TO_PATH] + MOVING_COST[dir]
                h = self._compute_euclidian_distance(next_point, goal)
                f = g + h

                next_costs = plan_grid.get_costs(next_point)

                if next_costs[F_CURRENT_BEST_GUESS] > f:
                    plan_grid.set_costs(next_point, [f, g, h])
                    plan_grid.set_parent(next_point, curr_point)
                    open_list.put(QueuedPoint(next_point, f))

        if found_goal is None:
            return None

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
        return path
