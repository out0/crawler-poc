import numpy as np
from jetson_inference import segNet
from jetson_utils import cudaImage, cudaAllocMapped, \
    cudaDeviceSynchronize, cudaToNumpy
from frame_segmenter import FrameSegmenter
from typing import Union


class Segmentation (FrameSegmenter):
    _mask_buffer: cudaImage = None
    _width: int
    _height: int
    _frame_format: str

    def __init__(self):
        self._width = 0
        self._height = 0
        self._frame_format = None
        self._mask_buffer = None

    def setupNetwork(self,
                     onnxFile: str,
                     labelFile: str,
                     colorFile: str,
                     networkWidth: int,
                     networkHeight: int,
                     inputLayerName: str,
                     outputLayerName: str):

        self._width = networkWidth
        self._height = networkHeight
        self._net = segNet(argv=[
            f"--model={onnxFile}",
            f"--labels={labelFile}",
            f"--colors={colorFile}",
            f"--input_blob={inputLayerName}",
            f"--output_blob={outputLayerName}",
        ])

    def segment(self, frame: cudaImage) -> Union[np.array, None]:
        if self._mask_buffer is None or self._frame_format != frame.format:
            self._mask_buffer = cudaAllocMapped(
                width=self._width,
                height=self._height,
                format=frame.format)

        self._net.Process(frame, ignore_class='void')
        self._net.Mask(self._mask_buffer, filter_mode='linear')
        cudaDeviceSynchronize()

        if self._mask_buffer is None:
            return None

        cudaToNumpy(self._mask_buffer)
