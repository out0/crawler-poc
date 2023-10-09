import paho.mqtt.client as paho
import threading
from typing import Any
import time

class MqttClient:
    _mqtt_client: paho.Client = None
    _callbacks = None
    _run: bool

    def __init__(self, broker_ip: str, broker_port: int) -> None:
        self._run = True
        self._mqtt_client = paho.Client(clean_session=True)
        self._mqtt_client.on_message = self._on_receive
        self._callbacks = {}
        self._mqtt_thr = threading.Thread(target=self.__mqtt_client_loop)
        self._mqtt_thr.start()
        self._mqtt_client.connect(broker_ip, broker_port, 3600)


    def __mqtt_client_loop(self):
        while(self._run):
            self._mqtt_client.loop()

    def subscribeTo(self, topic:str, callback:callable) -> bool:
        self._callbacks[topic] = callback
        timeout = 200
        while (not self._mqtt_client.is_connected() and timeout > 0):
            time.sleep(0.01)
            timeout = timeout - 1
        
        if timeout == 0:
            return False
    
        self._mqtt_client.subscribe(topic)
        return True
        

    def _on_receive(self, client, userdata, msg):
        if msg.topic not in self._callbacks.keys():
            return
        
        cb = self._callbacks[str(msg.topic)]
        cb(msg.payload.decode())

    def publishTo(self, topic: str, payload: str) -> None:
        self._mqtt_client.publish(topic, payload, retain=False)

    def disconnect(self):
        self._mqtt_client.disconnect()
        self._run = False