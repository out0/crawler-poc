import carla
from carlasim.mqtt_client import MqttClient
import json, time, threading
import paho
from carlasim.ego_car import EgoCar

class EgoCarRemoteController:
    _mqtt_client: MqttClient = None
    _ego_car: EgoCar

    def __init__(self, mqtt_client: MqttClient, ego_car: EgoCar) -> None:
        self._mqtt_client = mqtt_client
        self._ego_car = ego_car
        mqtt_client.subscribeTo("/virtual_car/action", self._action_process)

    def _action_process(self, payload):
        action = json.loads(payload)
        if action["action"] == "engine_power":
            self._ego_car.set_autonomous_mode_flag(False)
            self._ego_car.set_power(int(action["value"]))
            

        elif action["action"] == "steering_angle":
            self._ego_car.set_autonomous_mode_flag(False)
            self._ego_car.set_steering(int(action["value"]))

        elif action["action"] == "resume_autonomous_driving":
            self._ego_car.set_autonomous_mode_flag(int(action["value"]) == 1)

    
    def destroy(self) -> None:
        self._mqtt_client.unsubscribeFrom("/virtual_car/action")