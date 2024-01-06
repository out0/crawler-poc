#! /usr/bin/python3
import os, threading, time, numpy as np, math
from carlasim.carla_client import CarlaClient
from carlasim.ego_car import EgoCar
from carlasim.ego_car_remote_controller import EgoCarRemoteController
from carlasim.mqtt_client import MqttClient
from carlasim.sensors.carla_camera import *
from carlasim.simulation_log import *

import gi
from gi.repository import Gst
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
Gst.init(None)

MISSION_FILE = "mission.dat"

class SimulationController:
    __carla_client: CarlaClient
    __ego_car: EgoCar
    __manual_control: EgoCarRemoteController
    __manual_control_status: str
    __mqtt_client: MqttClient
    __autopilot_status: str
    __front_camera_status: str
    __bev_camera_status: str
    _logger: AutoSimulationLogger

    def __init__(self) -> None:
        self.__carla_client = CarlaClient(town='Town07')
        self.__ego_car = EgoCar(self.__carla_client)
        self.__manual_control_status = "OFF"
        self.__autopilot_status = "OFF"
        self.__manual_control = None
        self.__mqtt_client = None
        self.__front_camera_status = "OFF"
        self.__bev_camera_status = "OFF"
        self._logger = None  
        pass

    def __initialize_ego_car(self) -> None:
        self.__ego_car.set_pose(-100, 0, 0, 0)
        self.__ego_car.set_power(0)
        self.__ego_car.set_steering(0)

    def run (self) -> None:
        self.__initialize_ego_car()
        self.__run_menu()
    
    def __clear_input(self, value: str) -> str:
        return value.lower()
    
    
    def __get_option(self) -> int:
        if self._logger is None:
            auto_pose_log_status = "OFF"
        else:
            auto_pose_log_status = "ON"

        os.system("clear")

        print ("1. reset")
        print (f"2. toggle manual control [{self.__manual_control_status}]")
        print (f"3. toggle autopilot [{self.__autopilot_status}]")
        print ("4. run mission file")
        print ("5. log current pose")
        print (f"6. auto pose log [{auto_pose_log_status}]")
        print ("7. cameras")
        print ("8. print gstreamer")
        print ("x. exit")
        opt = self.__clear_input(input())
        if opt == "x":
            return 0
        try:
            return int(opt)
        except:
            return -1

    def __run_menu(self) -> None:
        while True:
            opt = self.__get_option()
            if opt < 0:
                continue
            elif opt == 0:
                break
            elif opt == 1:
                self.__stop_background_execution()
                self.__initialize_ego_car()
            elif opt == 2:
                self.__toggle_manual_control()
            elif opt == 3:
                self.__toggle_autopilot()                
            elif opt == 4:
                self.__execute_mission()
            elif opt == 5:
                self.__log_current_pose()
            elif opt == 6:
                self.__log_current_pose_periodically()
            elif opt == 7:
                self.__cameras_menu()
            elif opt == 8:
                for i in range(0, 3):
                    print (f'gst-launch-1.0 -v udpsrc port={20000+i} caps = \"application/x-rtp, media=\(string\)video, clock-rate=\(int\)90000, encoding-name=\(string\)H264, payload=\(int\)96\" ! rtph264depay ! decodebin ! videoconvert ! autovideosink\n')
                input("enter to return")

        self.__ego_car.destroy()
        if self.__mqtt_client is not None:
            self.__mqtt_client.disconnect()

    def __stop_background_execution(self) -> None:
        self.__ego_car.set_autopilot(False)
        self.__set_manual_control(False)
        if self._logger is not None:
            self._logger.destroy()
            self._logger = None
    
    def __set_manual_control(self, status: bool) -> None:
        if status:
            self.__mqtt_client = MqttClient("127.0.0.1", 1883)
            self.__manual_control = EgoCarRemoteController(self.__mqtt_client, self.__ego_car)
            self.__manual_control_status = "ON"
        else:
            if self.__manual_control is not None:
                self.__manual_control.destroy()
                self.__mqtt_client.disconnect()
            self.__manual_control = None
            self.__mqtt_client = None
            self.__manual_control_status = "OFF"
    
    def __set_auto_pilot(self, status: bool) -> None:
        if status:
            self.__autopilot_status = "ON"
            self.__ego_car.set_autopilot(True)
        else:
            self.__autopilot_status = "OFF"
            self.__ego_car.set_autopilot(False)

    def __toggle_manual_control(self) -> None:
        if self.__manual_control is None:
            self.__set_manual_control(True)
            self.__set_auto_pilot(False)     
        else:
            self.__set_manual_control(False)

    def __toggle_autopilot(self) -> None:
        if self.__autopilot_status == "OFF":
            self.__set_manual_control(False)
            self.__set_auto_pilot(True)
        else:
            self.__set_auto_pilot(False)       

    def __deactivate_front_camera_streaming(self) -> None:
            self.__front_camera_status = "OFF"
            self.__ego_car.stop_stream_front_camera()
            if self.__ego_car.front_camera is not None:
                self.__ego_car.front_camera.destroy()
            self.__ego_car.front_camera = None

    def __deactivate_bev_camera_streaming(self) -> None:
            self.__bev_camera_status = "OFF"
            self.__ego_car.stop_stream_bev_camera()
            if self.__ego_car.bev_camera is not None:
                self.__ego_car.bev_camera.destroy()
            self.__ego_car.bev_camera = BEVSemanticCamera(400, 300, 30)

    def __cameras_menu(self) -> None:
        print (f"Stream BEV camera: {self.__bev_camera_status}")
        if self.__bev_camera_status == "ON":
            print ("1, 2 or 3 for turning it OFF")
        else:
            print ("    1. RGB")
            print ("    2. Segmented / Colored")
            print ("    3. Segmented / Original")
        print ("")
        print (f"Stream front camera: {self.__front_camera_status}")
        if self.__front_camera_status == "ON":
            print ("4, 5 or 6 for turning it OFF")
        else:
            print ("    4. RGB")
            print ("    5. Segmented / Colored")
            print ("    6. Segmented / Original")

        opt = self.__clear_input(input())
        try:
            opt = int(opt)
        except:
            return
        
        f: CarlaCamera = None
        b: CarlaCamera = None

        if opt == 1:
            if self.__bev_camera_status == "ON":
                self.__deactivate_bev_camera_streaming()
                return                
            else:
                b = BEVRGBCamera(400, 300, 30)
                self.__bev_camera_status = "ON"

        elif opt == 2:
            if self.__bev_camera_status == "ON":
                self.__deactivate_bev_camera_streaming()
                return
            else:
                b = BEVColoredSemanticCamera(400, 300, 30)
                self.__bev_camera_status = "ON"            
            
        elif opt == 3:
            if self.__bev_camera_status == "ON":
                self.__deactivate_bev_camera_streaming()
                return
            else:
                b = BEVSemanticCamera(400, 300, 30)
                self.__bev_camera_status = "ON"    
        
        if opt == 4:
            if self.__front_camera_status == "ON":
                self.__deactivate_front_camera_streaming()
                return
            else:
                f = FrontRGBCamera(800, 600, 120, 30)
                self.__front_camera_status = "ON"

        elif opt == 5:
            if self.__front_camera_status == "ON":
                self.__deactivate_front_camera_streaming()
                return
            else:
                f = FrontColoredSemanticCamera(800, 600, 120, 30)
                self.__front_camera_status = "ON"            
            
        elif opt == 6:
            if self.__front_camera_status == "ON":
                self.__deactivate_front_camera_streaming()
                return
            else:
                f = FrontSemanticCamera(800, 600, 120, 30)
                self.__front_camera_status = "ON"   


        if f is not None and self.__ego_car.front_camera is not None:
            self.__ego_car.front_camera.destroy()
            self.__ego_car.front_camera = f
            f.attach_to(self.__carla_client, self.__ego_car.get_carla_ego_car_obj())
            self.__ego_car.stream_front_camera_to("127.0.0.1", 20001)

        if b is not None and self.__ego_car.bev_camera is not None:
            self.__ego_car.bev_camera.destroy()
            self.__ego_car.bev_camera = b
            b.attach_to(self.__carla_client, self.__ego_car.get_carla_ego_car_obj())
            self.__ego_car.stream_bev_camera_to("127.0.0.1", 20000)

    def __execute_mission(self) -> None:
        if not os.path.exists(MISSION_FILE):
            print (f"mission file {MISSION_FILE} not found")
            return
        
        waypoints = self.__read_mission(MISSION_FILE)

    def __log_current_pose(self) -> None:
        SimulationLogger(self.__ego_car, MISSION_FILE).log_pose()
        

    def __log_current_pose_periodically(self) -> None:

        if self._logger == None:
            print ("")
            print ("1. auto-log by distance")
            print ("2. auto-log by time span")
            print ("x. cancel")
            print ("")
            opt = self.__clear_input(input())
            if opt == "x":
                return
            elif opt == '1':
                dist = input("distance (m): ")
                try:
                    dist = int(dist)
                except:
                    return
                self._logger = AutoSimulationLogger(self.__ego_car, MISSION_FILE, trigger_dist=dist)

            elif opt == '2':
                time = input("time (ms): ")
                try:
                    time = int(time)
                except:
                    return
                self._logger = AutoSimulationLogger(self.__ego_car, MISSION_FILE, trigger_time_ms=time)
        else:
            self._logger.destroy()
            self._logger = None
    
def main():
    controller = SimulationController()
    controller.run()

if __name__ == '__main__':
    main()
    