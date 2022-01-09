from __future__ import absolute_import
from __future__ import print_function

import os.path as osp


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def __init__(self, root):
        self.root = osp.expanduser(root)

    def get_imagedata_info(self, data):
        pids = []
        for _, pid in data:
            pids += [pid]
            
        pids = set(pids)  
        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        #num_val_pids, num_val_imgs, num_val_cams = self.get_imagedata_info(val)
        num_query_pids, num_query_imgs = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs = self.get_imagedata_info(gallery)

        print('Image Dataset statistics:')
        print('  ----------------------------')
        print('  subset   | # ids | # images ')
        print('  ----------------------------')
        print('  train    | {:5d} | {:8d} '.format(num_train_pids, num_train_imgs))
        #print('  val      | {:5d} | {:8d} | {:9d}'.format(num_val_pids, num_val_imgs, num_val_cams))
        print('  query    | {:5d} | {:8d} '.format(num_query_pids, num_query_imgs))
        print('  gallery  | {:5d} | {:8d} '.format(num_gallery_pids, num_gallery_imgs))
        print('  ----------------------------')
