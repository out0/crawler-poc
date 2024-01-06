class Waypoint:
    x: int
    z: int

    def __init__(self, x: int, z: int):
        self.x = int(x)
        self.z = int(z)

    def __str__(self):
        return f"({self.x}, {self.z})"

class DistWaypoint(Waypoint):
    x: int
    z: int
    dist: float

    def __init__(self, x: int, z: int, dist: float):
        super().__init__(x, z)
        self.dist = dist

    def __str__(self):
        return f"({self.x}, {self.z}, dist: {self.dist})"