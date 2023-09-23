
import cv2
import numpy as np

class OutputWriter:
    _original: cv2.VideoWriter
    _segmented: cv2.VideoWriter
    _planned: cv2.VideoWriter

    def __init__(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._original = cv2.VideoWriter(
            'original.mp4', fourcc, 30.0, (1920, 1440))
        self._segmented = cv2.VideoWriter(
            'segmented.mp4', fourcc, 30.0, (352, 288))
        self._planned = cv2.VideoWriter('planned.mp4', fourcc, 30.0, (720, 960))

    def on_original_frame(self, frame: np.array) -> None:
        self._original.write(frame)

    def on_segmented_frame(self, frame: np.array) -> None:
        self._segmented.write(frame)

    def on_planned_frame(self, frame: np.array) -> None:
        self._planned.write(frame)

    def close(self):
        self._original.release()
        self._segmented.release()
        self._planned.release()
