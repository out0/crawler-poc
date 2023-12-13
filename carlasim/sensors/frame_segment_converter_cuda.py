import numpy as np
from numba import cuda, jit

class FrameSegmentConverterCuda:
    _cuda_colors: any
    
    def __init__(self) -> None:
        segmented_color = np.array([
            [0,   0,   0],
            [128,  64, 128],
            [244,  35, 232],
            [70,  70,  70],
            [102, 102, 156],
            [190, 153, 153],
            [153, 153, 153],
            [250, 170,  30],
            [220, 220,   0],
            [107, 142,  35],
            [152, 251, 152],
            [70, 130, 180],
            [220,  20,  60],
            [255,   0,   0],
            [0,   0, 142],
            [0,   0,  70],
            [0,  60, 100],
            [0,  80, 100],
            [0,   0, 230],
            [119,  11,  32],
            [110, 190, 160],
            [170, 120,  50],
            [55,  90,  80],
            [45,  60, 150],
            [157, 234,  50],
            [81,   0,  81],
            [150, 100, 100],
            [230, 150, 140],
            [180, 165, 180]
        ])

        self._cuda_colors = cuda.to_device(segmented_color)

    @cuda.jit
    def switch_values(image, switch_array):
        x, y = cuda.grid(2)  # 2D grid

        if x < image.shape[1] and y < image.shape[0]:
            channel_value = int(image[y, x, 0])
            if 0 <= channel_value < 20:
                for c in range(3):
                    image[y, x, c] = switch_array[channel_value, c]
    
    def convert_frame(self, frame) -> np.array:
        d_frame = cuda.to_device(np.ascontiguousarray(frame))
        threadsperblock = (16, 16)
        blockspergrid_x = (frame.shape[1] - 1) // threadsperblock[0] + 1
        blockspergrid_y = (frame.shape[0] - 1) // threadsperblock[1] + 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        FrameSegmentConverterCuda.switch_values[blockspergrid, threadsperblock](d_frame, self._cuda_colors)
        return d_frame.copy_to_host()