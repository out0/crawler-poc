from gi.repository import Gst
import threading
import queue
from carlasim.sensors.carla_camera import CarlaCamera
import gi
import numpy as np
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')


class VideoStreamer:
    _pipeline: any
    _appsrc: any
    _duration: int
    _is_streaming: bool
    _camera: CarlaCamera
    _launch_string: str
    _frame_count: int
    _frame_queue: queue.Queue
    _thr_stream: threading.Thread

    def _to_bgra_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a BGRA numpy array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def to_rgb_array(image) -> np.ndarray:
        """Convert a CARLA raw image to a RGB numpy array."""
        array = VideoStreamer._to_bgra_array(image)
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array

    def _ndarray_to_gst_buffer(array: np.ndarray) -> Gst.Buffer:
        """Converts numpy array to Gst.Buffer"""
        return Gst.Buffer.new_wrapped(array.tobytes())

    def __init__(self,  orig_width: int, orig_height: int, width: int, height: int, fps: int, ip: str, port: int) -> None:

        self._launch_string = f"appsrc name=source is-live=true format=time" \
            f" caps=video/x-raw,format=RGB,width={orig_width},height={orig_height},framerate={fps}/1 "   \
            f"! queue ! videoconvert ! videoscale ! video/x-raw,width={width},height={height}  ! x264enc ! " \
            f"rtph264pay ! queue ! udpsink host={ip} port={port} sync=false"

        self._fps = fps
        self._is_streaming = False
        self._frame_count = 0
        self._thr_stream = None
        self._frame_queue = None


    def start(self):
        try:
            self._pipeline = Gst.parse_launch(self._launch_string)
        except ValueError as ex:
            print(ex)
            raise ex

        self._appsrc = self._pipeline.get_child_by_name('source')
        self._duration = (1 / self._fps) * Gst.SECOND
        self._pipeline.set_state(Gst.State.PLAYING)
        

        self._frame_queue = queue.Queue()
        self._is_streaming = True
        self._thr_autostream = threading.Thread(None, self.__perform_streaming)
        self._thr_autostream.start()
        

    def stop(self):
        self._is_streaming = False
        if self._thr_stream is not None:
            self._thr_stream.join()
            self._thr_stream = None
        self._frame_queue = None

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsrc = None

    def __perform_streaming(self):
        while self._is_streaming:
            if self._frame_queue is None or self._frame_queue.empty():
                continue
            try:
                frame = self._frame_queue.get(block=False)
            except:
                continue

            if frame is None:
                continue

            self.__stream_frame(frame)

    def __stream_frame(self, frame: any):
        if not self._is_streaming:
            return

        if type(frame).__name__ != 'ndarray':
            frame = VideoStreamer.to_rgb_array(frame)

        buffer = VideoStreamer._ndarray_to_gst_buffer(frame)
        buffer.timestamp = Gst.CLOCK_TIME_NONE

        ret = self._appsrc.emit('push-buffer', buffer)
        if ret != Gst.FlowReturn.OK:
            print(ret)
            return

        if self._frame_count % self._fps == 0:
            event = Gst.Event.new_seek(1.0, Gst.Format.TIME, Gst.SeekFlags.FLUSH |
                                       Gst.SeekFlags.KEY_UNIT, Gst.SeekType.NONE, 0, Gst.SeekType.NONE, 0)
            self._pipeline.send_event(event)
            self._frame_count = 0

        self._frame_count += 1

    def new_frame(self, frame):
        if self._is_streaming:
            self._frame_queue.put(frame)
