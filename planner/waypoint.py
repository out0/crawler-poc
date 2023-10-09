class Waypoint:
    x: int
    z: int

    def __init__(self, x: int, z: int):
        self.x = int(x)
        self.z = int(z)

    def __str__(self):
        return f"({self.x}, {self.z})"