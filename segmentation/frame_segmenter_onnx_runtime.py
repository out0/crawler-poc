from .frame_segmenter import FrameSegmenter

# need to import anyway
import torch
# need to import anyway
import tensorrt
import onnxruntime as rt
import numpy as np
import cv2
from misc.label_colorizer import LabelColorizerWithBg


class FrameSegmenterOnnxRuntimeImpl (FrameSegmenter):

    def __init__(self, modelpath):

        self.colorizer = LabelColorizerWithBg(n_classes=7)
        self.modelpath = modelpath
        self.session = rt.InferenceSession(modelpath, providers=[
                                           'TensorrtExecutionProvider',
                                           'CUDAExecutionProvider',
                                           'CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mean = np.array([.485, .456, .406])
        self.std = np.array([.229, .224, .225])

    def _preprocess(self, x):
        x = x / 255
        x = cv2.resize(x, (352, 288))
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = x[None].astype(np.float32)
        return x

    def _predict(self, frame):
        x = self._preprocess(frame)
        return self.session.run(
            [self.output_name], {self.input_name: x})[0][0]

    def _predict_class(self, x):
        pred = self._predict(x)
        pred = pred.argmax(axis=0)
        return pred

    def segment(self, frame: any) -> any:
        pred = self._predict_class(frame)
        if pred is None:
            return None
        return (self.colorizer(pred) * 255).astype('uint8')
