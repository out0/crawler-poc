import numpy as np

class FrameSegmentConverter:
    width: int
    height: int

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self._segmented_color = np.array([
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

    def convert_frame(self, frame) -> None:
        for i in range(0, self.height - 1):
            for j in range(0, self.width - 1):
                if frame[i][j][0] > len(self._segmented_color):
                    colors = np.array([0, 0, 0])
                else:
                    colors = self._segmented_color[frame[i][j][0]]
                
                frame[i][j][0] = colors[0]
                frame[i][j][1] = colors[1]
                frame[i][j][2] = colors[2]

    def convert_clone_frame(self, frame, width: int, height: int) -> None:
        new_frame = np.zeros([height, width, 3], dtype='uint8')
        for i in range(0, height - 1):
            for j in range(0, width - 1):
                if frame[i][j][0] > len(self._segmented_color):
                    colors = np.array([0, 0, 0])
                else:
                    colors = self._segmented_color[frame[i][j][0]]

                new_frame[i][j][0] = colors[0]
                new_frame[i][j][1] = colors[1]
                new_frame[i][j][2] = colors[2]
        return new_frame
