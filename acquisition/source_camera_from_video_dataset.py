from jetson_utils import videoSource
from .source_camera import SourceCamera
import jetson.utils


class SourceCameraVideoDataset (SourceCamera):
    _input: videoSource

    def __init__(self, file: str):
        self._input = videoSource(file)
        pass

    def width(self) -> int:
        return self._input.GetWidth()

    def height(self) -> int:
        return self._input.GetHeight()

    def frameRate(self) -> int:
        return self._input.GetFrameRate()

    def capture(self) -> jetson.utils.cudaImage:
        return self._input.Capture()

    def isStreaming(self) -> bool:
        return self._input.IsStreaming()
