import numpy as np
import seaborn as sns
import torch


class LabelColorizer:

    def __init__(self, n_classes=12, ix_nolabel=255):
        assert isinstance(ix_nolabel, (int, type(None)))
        self.ix_nolabel = ix_nolabel
        self.n_classes = n_classes
        self.map = self.get_pallete()

    def get_pallete(self):
        pal = sns.color_palette(palette='gist_rainbow', as_cmap=True)(
            np.linspace(0, 1, self.n_classes))[..., :3]
        pal = np.vstack([[[0, 0, 0]], pal])
        dict_pal = {}
        for i in range(1, pal.shape[0]):
            dict_pal[i - 1] = torch.tensor(pal[i])

        if self.ix_nolabel is not None:
            dict_pal[self.ix_nolabel] = torch.tensor(pal[0])
        return dict_pal

    def __call__(self, mask):
        mask = torch.tensor(mask)[None]

        fl_single = False
        if len(mask.shape) < 4:
            fl_single = True
            mask = mask[None]

        bs = mask.shape[0]
        cmask = torch.zeros((bs, 3,) + mask.shape[2:]).type(self.map[1].dtype)
        for i in range(bs):
            for k in self.map.keys():
                cmask[i, :, mask[i, 0] == k] = self.map[k][:, None]

        if fl_single:
            cmask = cmask[0]

        cmask = cmask.permute(1, 2, 0).numpy()
        return cmask

    def reverse(self, x):
        mask_new = torch.zeros(x.shape[1:])
        for k, v in self.map.items():
            v_tensor = torch.Tensor(v)[:, None, None]
            k_pos = torch.all(torch.eq(x.float(), v_tensor), axis=0)
            mask_new[k_pos] = k


class LabelColorizerWithBg(LabelColorizer):

    def __init__(self, n_classes=12, **kwargs):
        super().__init__(n_classes, **kwargs)
        self.map = self.get_pallete()

    def get_pallete(self):
        pal = sns.color_palette(palette='gist_rainbow', as_cmap=True)(
            np.linspace(0, 1, self.n_classes - 1))[..., :3]
        pal = np.vstack([[[0, 0, 0]], pal])
        dict_pal = {}
        for i in range(pal.shape[0]):
            dict_pal[i] = torch.tensor(pal[i])
        return dict_pal
