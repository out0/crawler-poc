from carlasim.vehicle_hal import EgoCar
from threading import Thread
import time

class VelocityController:
    _factor: float
    _adder_thr: Thread
    _error_tolerance: float
    _target_value: float
    _is_keeping_value: bool
    _current_actuator_value: float
    _quick_start_vaule: float


    def __init__(self, car: EgoCar,  error_tolerance: float) -> None:
        self._car = car
        self._factor = 10.0
        self._error_tolerance = error_tolerance
        self._is_running = False
        self._target_value = -1
        self._is_keeping_value = False
        self._adder_thr = None
        self._current_actuator_value = 0
        self._quick_start_vaule  = 100 
    
    def set_speed(self, value: float) -> None:
        self._is_keeping_value = False
        self._adder_thr = Thread(target=self.__adder_func)
        self._target_value = value        
        self._adder_thr.start()
    
    def get_value(self) -> float:
        return self._measurer()
    
    def stop(self) -> None:
        self._is_keeping_value = False

    def __driving_actuator(self, val: int) -> None:
        self._car.set_engine_power(val)
        time.sleep(0.5)

    def __get_current_speed(self) -> float:
        return self._car.get_speed()
    

    def __quick_start(self) -> None:
        if self._quick_start_vaule <= 0:
            return

        current_value = self.__get_current_speed()
        dv = abs(current_value - self._target_value)
        if current_value == 0 and dv > 20:
            self._current_actuator_value =  self._quick_start_vaule

    def __adder_func(self):
        self._is_keeping_value = True
        self.__quick_start()

        while self._is_keeping_value:
            current_value = self.__get_current_speed()
            dv = abs(current_value - self._target_value)
            if current_value > self._target_value and dv > self._error_tolerance:
                self._current_actuator_value -= self._factor
                self.__driving_actuator(self._current_actuator_value)
            elif current_value < self._target_value and dv > self._error_tolerance:
                self._current_actuator_value += self._factor
                self.__driving_actuator(self._current_actuator_value)
