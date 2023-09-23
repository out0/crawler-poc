import numpy as np
import cv2


class BEVProjector:

    def __init__(
        self, grid_dims=(720, 960), edge_size=1e-2, uv_dims=(1440, 1920)
    ):

        self.R = np.array([
            [0.99792479, 0.02269552, -0.06025794],
            [-0.03158676, 0.98803323, -0.15097222],
            [0.05611045, 0.15256228,  0.98669974]
        ])

        self.tvec = np.array([
            [0.023667],
            [0.34899726],
            [0.91836986]
        ])

        self.mtx = np.array([
            [967.24986511,   0., 981.31348404],
            [0., 988.70685899, 728.97850079],
            [0.,   0.,   1.]
        ])

        self.dist = np.array([
            [0.00371372, -0.00157878,  0.00621299,  0.00757212,  0.00170795]])

        self.uv_dims = uv_dims  # h x w
        self.grid_dims = grid_dims
        self.edge_size = edge_size
        self._compute_uvcoords()

    def _compute_uvcoords(self):
        cam_offset_Z = (-self.tvec[[2]] + 10e-2) / self.edge_size
        cam_offset_X = (-self.tvec[[0]]) / self.edge_size

        og_grid_x = np.arange(
            - self.grid_dims[0] // 2 + cam_offset_X,
            (self.grid_dims[0] + .999) // 2 + cam_offset_X, 1)

        og_grid_z = np.arange(
            self.grid_dims[1] + cam_offset_Z - 1, cam_offset_Z - .999, -1)

        og_grid = \
            np.concatenate(
                np.meshgrid(
                    og_grid_x,
                    0,  # I want only the floor information, so height = 0
                    og_grid_z,
                    1
                )
            )

        og_grid = np.moveaxis(og_grid, 0, 2)
        og_grid_3dcoords = og_grid.copy()
        og_grid_3dcoords[:, :, :3, :] *= self.edge_size

        # Transform the world coords to cam coords
        RT = np.eye(4)
        RT[:3, :3] = self.R
        RT[:3, 3] = self.tvec.ravel()
        og_grid_camcoords = (RT @ og_grid_3dcoords.reshape(-1, 4).T)
        og_grid_camcoords = og_grid_camcoords.T.reshape(self.grid_dims + (4,))
        og_grid_camcoords /= og_grid_camcoords[..., [2]]
        og_grid_camcoords = og_grid_camcoords[..., :3]

        # Transform the cam coords to image coords
        og_grid_uvcoords = (self.mtx @ og_grid_camcoords.reshape(-1, 3).T)
        og_grid_uvcoords = og_grid_uvcoords.T.reshape(self.grid_dims + (3,))
        og_grid_uvcoords = og_grid_uvcoords.round().astype(int)
        og_grid_uvcoords = og_grid_uvcoords[..., :2]
        og_grid_uvcoords = np.moveaxis(og_grid_uvcoords, 0, 1)

        og_grid_uvcoords[..., 1] = og_grid_uvcoords[..., 1].clip(
            0, self.uv_dims[0])
        og_grid_uvcoords[..., 0] = og_grid_uvcoords[..., 0].clip(
            0, self.uv_dims[1])
        self.og_grid_uvcoords = og_grid_uvcoords

    def __call__(self, frame):
        bev_frame = cv2.resize(frame, (1920, 1440),
                               interpolation=cv2.INTER_NEAREST)
        x = np.pad(bev_frame, ((0, 1), (0, 1), (0, 0)), constant_values=0)
        return x[self.og_grid_uvcoords[..., 1], self.og_grid_uvcoords[..., 0]]
