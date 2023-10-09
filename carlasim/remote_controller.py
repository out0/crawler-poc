import carla
from .mqtt_client import MqttClient
import json
from carlasim.carla_sim_controller import CarlaSimulatorController


class RemoteController:
    _mqtt_client: MqttClient = None
    _target: any
    _vehicle_control: carla.VehicleControl
    _on_autonomous_driving_state_change: callable


    def __init__(self, mqtt_client: MqttClient, target: any, on_autonomous_driving_state_change: callable) -> None:
        self._mqtt_client = mqtt_client
        self._target = target
        self._vehicle_control = carla.VehicleControl()
        mqtt_client.subscribeTo("/virtual_car/action", self._action_process)
        self._on_autonomous_driving_state_change = on_autonomous_driving_state_change

    def _action_process(self, payload):
        action = json.loads(payload)
        if action["action"] == "engine_power":
            self._change_autonomous_driving_state(False)
            self._set_engine_power(int(action["value"]))
            

        elif action["action"] == "steering_angle":
            self._change_autonomous_driving_state(False)
            self._set_steering_angle(int(action["value"]))

        elif action["action"] == "resume_autonomous_driving":
            self._change_autonomous_driving_state(int(action["value"]))

    def _set_engine_power(self, power:int):
        
        self._vehicle_control.throttle = abs(power) / 240
        self._vehicle_control.reverse = power < 0
        self._vehicle_control.brake = 0.0
        self._target.apply_control(self._vehicle_control)
        

    def _set_steering_angle(self, degree:int):
        self._vehicle_control.steer = degree / 40
        self._target.apply_control(self._vehicle_control)
        
    def _change_autonomous_driving_state(self, state: bool):
        self._on_autonomous_driving_state_change(state)
        
