from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import *
from .data_loading import ImageDataset
from .init_dataset import init_imgreid_dataset
from .transform import test_transform


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 root='datasets',
                 height=128,
                 width=256,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 val_sampler='',
                 random_erase=False,  # use random erasing for data augmentation
                 color_jitter=False,  # randomly change the brightness, contrast and saturation
                 color_aug=False,  # randomly alter the intensities of RGB channels
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.source_names = source_names
        self.target_names = target_names
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances

        transform_test = test_transform(self.height, self.width)
        self.transform_test = transform_test


    def return_dataloaders(self):
        """
        Return testloader dictionary
        """
        return  self.testloader_dict

    def return_testdataset_by_name(self, name):
        """
        Return query and gallery, each containing a list of (img_path, pid, camid).
        """
        return self.testdataset_dict[name]['query'], self.testdataset_dict[name]['gallery']


class ImageDataManager(BaseDataManager):
    """
    Vehicle-ReID data manager
    """
    def __init__(self,
                 use_gpu,
                 source_names,
                 target_names,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, source_names, target_names, **kwargs)

        print('=> Initializing TEST (target) datasets')
        self.testloader_dict = {name: {'query': None, 'gallery': None} for name in target_names}
        self.testdataset_dict = {name: {'query': None, 'gallery': None} for name in target_names}

        for name in self.target_names:
            dataset = init_imgreid_dataset(
                root=self.root, name=name)

            self.testloader_dict[name]['query'] = DataLoader(
                ImageDataset(dataset.query, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testloader_dict[name]['gallery'] = DataLoader(
                ImageDataset(dataset.gallery, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

            self.testdataset_dict[name]['query'] = dataset.query
            self.testdataset_dict[name]['gallery'] = dataset.gallery
