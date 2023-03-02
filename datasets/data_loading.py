import pickle
import torch
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset

class VeriDataset(Dataset):
    """Veri dataset."""

    def __init__(self, pkl_file, dataset, root_dir, transform=None):
       
        with open(pkl_file, 'rb') as handle:
            c = pickle.load(handle)
        self.index = c
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.dataset[idx][0])
        img = Image.open(os.path.join(self.root_dir, img_name[-24:])).convert('RGB')
        label = self.dataset[idx][1]
        pid = self.dataset[idx][2]
        cid = self.dataset[idx][3]
        if self.dataset[idx][0] not in self.index:
            index = 0
        else:
            index = self.index[self.dataset[idx][0]][1]

        if self.transform:
            img = self.transform(img)

        return img,label,index,pid, cid

class IdDataset(Dataset):
    """VehicleId dataset."""

    def __init__(self, pkl_file, dataset, root_dir, transform=None):
       
        with open(pkl_file, 'rb') as handle:
            c = pickle.load(handle)
        self.index = c
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.dataset[idx][0])
        img = Image.open(os.path.join(self.root_dir, img_name[-17:])).convert('RGB')
        label = self.dataset[idx][1]
        pid = self.dataset[idx][2]
        cid = self.dataset[idx][3]
        index = self.index[self.dataset[idx][0]][1]
    

        if self.transform:
            img = self.transform(img)

        return img,label,index,pid, cid

class DukeDataset(Dataset):
    """Duke dataset."""

    def __init__(self, pkl_file, dataset, root_dir, transform=None):
       
        with open(pkl_file, 'rb') as handle:
            c = pickle.load(handle)
        self.index = c
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir,
                                self.dataset[idx][0])
        img = Image.open(os.path.join(self.root_dir, img_name[-20:])).convert('RGB')
        label = self.dataset[idx][1]
        pid = self.dataset[idx][2]
        cid = self.dataset[idx][3]
        index = self.index[self.dataset[idx][0]][1]
    

        if self.transform:
            img = self.transform(img)

        return img,label,index,pid, cid


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class IdImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, _, img_path
