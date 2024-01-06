
import numpy as np
from .discrete_component import DiscreteComponent

class LongitudinalController(DiscreteComponent):
    __KP = 1.0
    __KI = 0.2
    __KD = 0.01

    _odometer: callable
    _power_actuator : callable
    _brake_actuator : callable
    _error_prev: float
    _error_I: float
    _error_D: float
    _desired_speed: float
    _prev_throttle: float
   
    def __init__(self, sampling_period_ms: int, odometer: callable, power_actuator: callable, brake_actuator: callable) -> None:
        super().__init__(sampling_period_ms)
        self._desired_speed = 0.0
        self._prev_throttle = 0.0
        self._odometer = odometer
        self._power_actuator = power_actuator
        self._brake_actuator = brake_actuator
        self._error_I = 0.0
        self._error_D = 0.0
        self._error_prev = 0.0

    def brake(self, brake_strenght: float) -> None:
        self._power_actuator(0)
        self._brake_actuator(brake_strenght)
    
    def _loop(self, dt: float) -> None:
        current_speed = self._odometer()
        error = self._desired_speed - current_speed

        # Autobreak
        if current_speed == 0 and self._desired_speed == 0:
            self.brake(1.0)
            return

        # print(f"dt = {dt}")
        self._error_I += error * dt
        self._error_D = (error - self._error_prev) / dt
        acc = LongitudinalController.__KP * error \
                    + LongitudinalController.__KI * self._error_I\
                    + LongitudinalController.__KD * self._error_D
        
        # print (f"err = {error}")
        # print (f"nerr_I = {self._error_I}")
        # print (f"err_D = {self._error_D}")
        
        self._error_prev = error

    #    if acc < 0:
    #        self.brake(0.5)
    #        return

    #    if acc == 0:
    #        self.brake(0.0)
    #        return
        
        throttle = (np.tanh(acc) + 1)/2
        if throttle - self._prev_throttle > 0.1:
            throttle = self._prev_throttle + 0.1
        
        self._prev_throttle = throttle      
        self._power_actuator(240 * throttle)

        # print (f"throttle = {throttle}")
        # print (f"speed = {current_speed}")
       
      

    def set_speed(self, desired_speed: float) -> None:
        self._desired_speed = desired_speed

    def destroy(self) -> None:
        if self._desired_speed != 0:
            self._desired_speed = 0.0

        self.brake(1.0)
        super().destroy()