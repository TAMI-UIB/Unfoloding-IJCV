import numpy as np
import torch
from torch.nn.functional import fold, unfold


class CreatePatches():
    def __init__(self, data):
        self.patch_height = 200
        self.patch_width = 300
        self.C = data.size(0)
        self.H = data.size(1)
        self.W = data.size(2)

    def do_patches(self, data):
        data = data.unsqueeze(0)
        N = data.size(0)
        data_shape = data.size
        if not data_shape(2) % self.patch_height == 0 and not data_shape(3) % self.patch_width == 0:
            raise ValueError('Data shape must be divisible by patch size')

        patches = unfold(data, kernel_size=(self.patch_height, self.patch_width), stride=(self.patch_height, self.patch_width))
        patch_num = patches.shape[-1]
        patches = patches.permute(0, 2, 1).contiguous().view(N * patch_num, self.C, self.patch_height, self.patch_width)
        return patches.squeeze(0)

    def undo_patches(self, data):
        num_patches_per_img = (self.H // self.patch_height) * (self.W // self.patch_width)

        data = data.view(1, num_patches_per_img, self.C * self.patch_height * self.patch_width)
        data = data.permute(0, 2, 1)
        N = data.size(0)

        patches = data.view(N, num_patches_per_img, self.C * self.patch_height * self.patch_width)
        patches = patches.permute(0, 2, 1)

        reconstructed = fold(patches,
                               output_size=(self.H, self.W),
                               kernel_size=(self.patch_height, self.patch_width),
                               stride=(self.patch_height, self.patch_width))
        return reconstructed
