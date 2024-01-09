from model.vehicle_pose import VehiclePose
import math

class RangeResult:
    is_between_ref_points_in_path: bool
    estimated_dist: float
    proportion_to_p2: float

    def __init__(self,
                is_between_ref_points_in_path: bool,
                estimated_dist: float,
                proportion_to_p2: float) -> None:
        self.is_between_ref_points_in_path = is_between_ref_points_in_path
        self.estimated_dist = estimated_dist
        self.proportion_to_p2 = proportion_to_p2


class ReferencePath:
    _p1: VehiclePose
    _p2: VehiclePose

    def __init__(self, p1: VehiclePose, p2: VehiclePose) -> None:
        self._p1 = p1
        self._p2 = p2

    def update_path(self, p1: VehiclePose, p2: VehiclePose) -> None:
        self._p1 = p1
        self._p2 = p2

    def get_nearest_point(self, p: VehiclePose) -> VehiclePose:
        dx= self._p2.x - self._p1.x
        dy = self._p2.y - self._p1.y
        det = dx*dx + dy*dy
        a = (dy*(p.y - self._p1.y)+dx*(p.x - self._p1.x))/det
        return VehiclePose(self._p1.x + a*dx, self._p1.y + a*dy, self._p1.heading, 0)

    def distance_to_line(self, p: VehiclePose) -> float:
        dx = self._p2.x - self._p1.x
        dy = self._p2.y - self._p1.y
       
        num = dx*(self._p1.y - p.y) - (self._p1.x - p.x)*dy
        den = math.sqrt((dx ** 2 + dy ** 2))
        return num / den

    def __eucl_compute_heading(dx: float, dy: float) -> float:
        deg90 = math.pi / 2

        if dy >= 0 and dx > 0:
            return deg90 - math.atan(dy/dx)
        elif dy >= 0 and dx < 0:
            return math.atan(dy/abs(dx)) - deg90
        elif dy <= 0 and dx < 0:
            return -(math.atan(dy/dx) + deg90)
        elif dy <= 0 and dx > 0:
            return  math.atan(abs(dy)/dx) + deg90
        elif dx == 0 and dy > 0:
            return deg90
        elif dx == 0 and dy < 0:
            return -deg90
        return 0.0

    def compute_heading(self) -> float:
        dy = self._p2.y - self._p1.y
        dx = self._p2.x - self._p1.x

        # maps in carla are inverted
        return ReferencePath.__eucl_compute_heading(dy, dx)
    
    def compute_dbl_euclidian_dist(self, p1: VehiclePose, p2: VehiclePose) -> float:
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return dx ** 2 + dy ** 2
    
    def check_in_range(self, ego_pos: VehiclePose, max_range_squared: float) -> RangeResult:
        nearest = self.get_nearest_point(ego_pos)
        
        l =  math.floor(self.compute_dbl_euclidian_dist(self._p1, self._p2))

        dist_from_p1 = math.floor(self.compute_dbl_euclidian_dist(self._p1, nearest))
        
        dist_from_p2 =  math.floor(self.compute_dbl_euclidian_dist(self._p2, nearest))
        
        if dist_from_p1 > l or dist_from_p1 >= max_range_squared:
            return RangeResult(False, dist_from_p1, -1)
        
        if dist_from_p2 > l or dist_from_p2 >= max_range_squared:
            return RangeResult(False, dist_from_p2, -1)
        
        return RangeResult(True, dist_from_p2, dist_from_p2/l)