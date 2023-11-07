from carlasim.vehicle_hal import EgoCar
from threading import Thread
import time

class DirectionController:
    _car: EgoCar
    _controller_thr: Thread
    _running: bool
    _new_heading_value_degrees: int
    _error_tolerance: float

    def __init__(self, car: EgoCar, error_tolerance: float) -> None:
        self._car = car
        self._running = False
        self._heading_value_degrees = 0
        self._error_tolerance = error_tolerance
        pass
    

    def __compute_attacking_angle(self,  current_heading: float, new_heading: float):
        attacking_angle = abs(new_heading - current_heading)
        if attacking_angle > 20:
            attacking_angle = 20
        return attacking_angle

    def _heading_control(self) -> None:        
        self._running = True

        while self._running: 
            current_heading = self.__get_current_heading()
            attacking_angle = self.__compute_attacking_angle(current_heading, self._new_heading_value_degrees)
            if self._new_heading_value_degrees > current_heading:
                self._car.steer(attacking_angle)
            else:
                self._car.steer(-attacking_angle)
            if attacking_angle <= self._error_tolerance:
                self._running = False
            time.sleep(0.5)


    def set_heading(self, degrees: int) -> None:
        self.stop()
        self._new_heading_value_degrees = degrees
        self._controller_thr = Thread(target=self._heading_control)
        self._controller_thr.start()

    def stop(self) -> None:
        self._running = False

    def __get_current_heading(self) -> float:
        h = self._car.get_heading()
        #print (f"current heading: {h}")
        return h