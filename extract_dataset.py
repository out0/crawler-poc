#
# Dataset extraction from carla or video-file
#
# by out0

import sys
from typing import List
import cv2
import time
import queue
from carlasim.video_streamer import VideoStreamer
from planner.occupancy_grid import OccupancyGrid

class ExtractionConfig:
    file: str
    waiting_time_s: int
    recording_time_s: int
    extract_to_frames: bool
    carla_town: str

    def __init__(self) -> None:
        self.file = None
        self.waiting_time_s = 1000
        self.recording_time_s = 10000
        self.extract_to_frames = False
        self.carla_town = 'Town07'

class UserCmd:
    
    def _help(argv: List[str]):
        print (f"use {argv[0]}\n")
        print ("-v [video-file]\t\t\t\t- extracts from a video dataset")
        print ("-c <town> \t\t\t\t- extracts from the carla simulator. Optionally, you can specify the town for the simulation")
        print ("-w [waiting time before recording]\t- waits x seconds before starting to extract data")
        print ("-r [recording time]\t\t\t- extracts data for x seconds")
        print ("-f \t\t\t\t\t- extract to frames\n\n")
    
    def _check_next_argument(i: int, argc: int, error_msg: str) -> bool:
        if i == argc - 1:
            print (error_msg)
            return False
        return True        

    def proc_input(argc: int, argv: List[str]) -> ExtractionConfig:
        if argc < 2:
            UserCmd._help(argv)
            return None
        
        conf = ExtractionConfig()

        i = 1
        while i < argc:
            if argv[i] == '-v':
                if not UserCmd._check_next_argument(i, argc, "error: you must specify a file"):
                    return None
                i += 1
                conf.file = argv[i]
            elif argv[i] == '-c':
                conf.file = None
                if argc > i + 1 and argv[i+1][0] != '-':
                    i += 1
                    conf.carla_town = argv[i]
            elif argv[i] == '-w':
                if not UserCmd._check_next_argument(i, argc, "error: you must specify a value for the waiting time"):
                    return None
                try:
                    i += 1
                    conf.waiting_time_s = int(argv[i])
                except:
                     print ("error: invalid waiting time")
                     return None
            elif argv[i] == '-r':
                if not UserCmd._check_next_argument(i, argc, "error: you must specify a value for the recording time"):
                    return None
                try:
                    i += 1
                    conf.recording_time_s = int(argv[i])
                except:
                     print ("error: invalid recording time")
                     return None
            elif argv[i] == '-f':
                conf.extract_to_frames = True

            i += 1
        
        return conf

class OutputWriter:

    def write (self, frame: any) -> None:
        pass

    def close() -> None:
        pass

class OutputVideoWriter(OutputWriter):

    _video_writer: cv2.VideoWriter

    def __init__(self, video_writer: cv2.VideoWriter) -> None:
        super().__init__()
        self._video_writer = video_writer  

    def write (self, frame: any) -> None:
        self._video_writer.write(frame)

    def close(self) -> None:
        self._video_writer.release()

class OutputFrameWriter(OutputWriter):
    _prefix: str
    _path: str
    _frame_count: int

    def __init__(self, prefix: str, path: str) -> None:
        super().__init__()
        self._prefix = prefix
        self._path = path
        self._frame_count = 0

    def write (self, frame: any) -> None:
        cv2.imwrite(f"{self._path}/{self._prefix}_{self._frame_count}.png", frame)
        self._frame_count += 1

    def close(self) -> None:
        pass

def save_bev(can_output_data: bool, outp: OutputWriter, outp_color: OutputWriter, frame: any) -> None:
    f = VideoStreamer.to_rgb_array(frame)
    og = OccupancyGrid(f, 0, 0)
    if can_output_data:
        outp.write(f)
        outp_color.write(og.get_color_frame())

def save_orig(can_output_data: bool, outp: OutputWriter, frame: any) -> None:
    if can_output_data:
        outp.write( VideoStreamer.to_rgb_array(frame))

def extract_from_carla(conf: ExtractionConfig):
    from carlasim.carla_sim_controller import CarlaClient
    from carlasim.vehicle_hal import EgoCar
    
    print(f"setting up simulation to '{conf.carla_town}'")
    client = CarlaClient(conf.carla_town)   

    if conf.extract_to_frames:
        outp_orig = OutputFrameWriter("original", "results/carla")
        outp_bev = OutputFrameWriter("bev", "results/carla")
        outp_bev_color = OutputFrameWriter("bev_color", "results/carla")
    else:
        outp_orig = OutputVideoWriter(cv2.VideoWriter('results/carla/original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300)))
        outp_bev = OutputVideoWriter(cv2.VideoWriter('results/carla/bev.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300)))
        outp_bev_color = OutputVideoWriter(cv2.VideoWriter('results/carla/bev_color.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (400, 300)))

    can_output_data = False

    print("building ego car")
    car = EgoCar(client)\
        .with_rgb_camera(20000, lambda f: save_orig(can_output_data, outp_orig, f))\
        .with_bev_camera(20001, lambda f: save_bev(can_output_data, outp_bev, outp_bev_color, f))\
        .autopilot()\
        .build()
    
    #print("setting pose")
    #car.set_pose(0, 0, 0)
    
    print(f"waiting {conf.waiting_time_s}s to start recording")
    time.sleep(conf.waiting_time_s)
    can_output_data = True
    
    print(f"recording {conf.recording_time_s}s of simulation...")
    time.sleep(conf.recording_time_s)
    can_output_data = False

    outp_orig.close()
    outp_bev.close()
    outp_bev_color.close()

    car.destroy()

class SharedControlData:
    running: bool
    frame_queue: queue.Queue
    outp_seg: OutputWriter
    outp_bev: OutputWriter
    segmenter: any
    projector: any

    def __init__(self, frame_queue: queue.Queue, outp_seg: OutputWriter, outp_bev: OutputWriter, segmenter: any, projector: any) -> None:
        self.running = True
        self.frame_queue = frame_queue
        self.outp_seg = outp_seg
        self.outp_bev = outp_bev
        self.segmenter = segmenter
        self.projector = projector
       

def async_process_dataset_frame(ctrl: SharedControlData):
    while ctrl.running or not ctrl.frame_queue.empty():
        if ctrl.frame_queue.empty():
            continue
        try:
            f = ctrl.frame_queue.get(block=False)
            seg = ctrl.segmenter.segment(f)
            ctrl.outp_seg.write(seg)
            ctrl.outp_bev.write(ctrl.projector(seg))
        except:
            continue


def extract_from_dataset(conf: ExtractionConfig):
    from segmentation.frame_segmenter import FrameSegmenter
    from segmentation.frame_segmenter_onnx_runtime import FrameSegmenterOnnxRuntimeImpl
    from bev.bev_projector import BEVProjector
    import threading

    input = cv2.VideoCapture(conf.file)
    if not input.isOpened():
        print (f"cant open {conf.file}")
        return

    model = "/workspaces/crawler-poc/assets/rtkbosque.onnx"
    

    print("loading onnx model")
    segmenter: FrameSegmenter = FrameSegmenterOnnxRuntimeImpl(model)


    print("waiting")
    t1 = time.time()
    t2 = t1
    while t2 - t1 < conf.waiting_time_s:
        input.read()
        t2 = time.time()
    
    if not input.isOpened():
        print ("the video ended up unexpectedly")
        return
    
    if conf.extract_to_frames:
        outp_orig = OutputFrameWriter("original", "results/dataset")
        outp_seg = OutputFrameWriter("segmented", "results/dataset")
        outp_bev = OutputFrameWriter("bev", "results/dataset")
    else:
        outp_orig = OutputVideoWriter(cv2.VideoWriter('results/dataset/original.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (1920, 1440)))
        outp_seg = OutputVideoWriter(cv2.VideoWriter('results/dataset/segmented.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (352, 288)))
        outp_bev = OutputVideoWriter(cv2.VideoWriter('results/dataset/bev.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (720, 960)))
    projector = BEVProjector()

    ctrl = SharedControlData(queue.Queue(), outp_seg, outp_bev, segmenter, projector)

    proc_thr = threading.Thread(None, async_process_dataset_frame, name="proc", args=[ctrl])
    proc_thr.start()

    print("recording")
    t1 = time.time()
    t2 = t1
    while t2 - t1 < conf.recording_time_s:
        r, f = input.read()
        if r:
            outp_orig.write(f)
            ctrl.frame_queue.put(f)           
        t2 = time.time()

    ctrl.running = False
    proc_thr.join()

    outp_orig.close()
    outp_seg.close()
    outp_bev.close()

def execute_extract(conf: ExtractionConfig):
    if conf.file is None:
        extract_from_carla(conf)
    else:
        extract_from_dataset(conf)
    print("done")


if __name__ == "__main__":
    conf = UserCmd.proc_input(len(sys.argv), sys.argv)
    if conf is not None:
        execute_extract(conf)