import os

import torch
from PIL import Image
import torchvision.transforms as transforms
from data.patchwork import CreatePatches



class LOL_paired(torch.utils.data.Dataset):
    def __init__(self, path, type, subset):
        super(LOL_paired, self).__init__()
        self.path_h = f'{path}/{type}/{subset}/high'

        self.path_l = f'{path}/{type}/{subset}/low'
        self.image_list_h = []
        self.image_list_l = []
        high= os.listdir(self.path_h)
        low= os.listdir(self.path_l)
        high.sort()
        low.sort()
        self.to_tensor = transforms.ToTensor()
        self.gt_h = []
        self.gt_l = []
        self.L0_l = []
        self.R0_l = []
        self.epsilon = 0.01

        for file in high:
            if file.endswith('.png'):
                self.image_list_h.append(file)

        for name in self.image_list_h:
            img = Image.open(f'{self.path_h}/{name}').convert('RGB')
            tensor_image = self.to_tensor(img)
            self.gt_h.append(tensor_image)

        for file in low:
            if file.endswith('.png'):
                self.image_list_l.append(file)

        for name in self.image_list_l:
            img = Image.open(f'{self.path_l}/{name}').convert('RGB')
            tensor_imagel = self.to_tensor(img)
            self.gt_l.append(tensor_imagel)
        self.generate_initialLR_low()

    def __len__(self):
        return len(self.gt_h)

    def channels(self):
        return 3

    def __getitem__(self, index):
        return self.gt_h[index], self.gt_l[index], self.L0_l[index], self.R0_l[index]

    def generate_initialLR_low(self):
        for gt in self.gt_l:
            L = torch.max(gt, dim=0, keepdim=True).values
            self.L0_l.append(L)
            R = gt / (L + self.epsilon * torch.ones_like(L))
            self.R0_l.append(R)

class LOL_patches(torch.utils.data.Dataset):
    def __init__(self, path, type, subset):
        super(LOL_patches, self).__init__()
        self.path_h = f'{path}/{type}/{subset}/high'

        self.path_l = f'{path}/{type}/{subset}/low'
        self.image_list_h = []
        self.image_list_l = []
        high = os.listdir(self.path_h)
        low = os.listdir(self.path_l)
        high.sort()
        low.sort()
        self.to_tensor = transforms.ToTensor()
        self.gt_h = []
        self.gt_l = []
        self.L0_l = []
        self.R0_l = []
        self.epsilon = 0.01

        for file in high:
            if file.endswith('.png'):
                self.image_list_h.append(file)

        for name in self.image_list_h:
            img = Image.open(f'{self.path_h}/{name}').convert('RGB')
            tensor_image = self.to_tensor(img)
            gt_patcher = CreatePatches(tensor_image)
            tensor_image = gt_patcher.do_patches(tensor_image)
            self.gt_h.append(tensor_image)
        for file in low:
            if file.endswith('.png'):
                self.image_list_l.append(file)

        for name in self.image_list_l:
            img = Image.open(f'{self.path_l}/{name}').convert('RGB')
            tensor_imagel = self.to_tensor(img)
            low_patcher = CreatePatches(tensor_imagel)
            tensor_imagel = low_patcher.do_patches(tensor_imagel)
            self.gt_l.append(tensor_imagel)
            Lpatch = torch.max(tensor_imagel, dim=1, keepdim=True).values
            Rpatch = tensor_imagel / (Lpatch + self.epsilon * torch.ones_like(Lpatch))
            self.L0_l.append(Lpatch)
            self.R0_l.append(Rpatch)

    def __len__(self):
        return len(self.gt_h)

    def channels(self):
        return 3

    def __getitem__(self, index):
        return self.gt_h[index], self.gt_l[index], self.L0_l[index], self.R0_l[index]

    def generate_initialLR_low(self):
        for gt in self.gt_l:
            L = torch.max(gt, dim=0, keepdim=True).values
            self.L0_l.append(L)
            R = gt / (L + self.epsilon * torch.ones_like(L))
            self.R0_l.append(R)
