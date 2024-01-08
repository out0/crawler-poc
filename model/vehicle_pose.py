import math

class VehiclePose:
    x: float
    y: float
    heading: float
    desired_speed: float


    def __init__(self, x: float, y: float, heading_angle: float, desired_speed: float):
        self.x = x
        self.y = y
        self.heading = heading_angle
        self.desired_speed = desired_speed

    def __str__(self):
        return f"({self.x}, {self.y}, heading: {self.heading})"
    

    def distance_to(self, pose : 'VehiclePose') -> float:
        dx = pose.x - self.x
        dy = pose.y - self.y
        return math.sqrt(dx*dx + dy*dy) 
    