from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os
import os.path as osp
import glob
import re

from .base import BaseImageDataset
from collections import defaultdict


class VehicleID(BaseImageDataset):
    """
    VehicleID

    Reference:
    @inproceedings{liu2016deep,
    title={Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles},
    author={Liu, Hongye and Tian, Yonghong and Wang, Yaowei and Pang, Lu and Huang, Tiejun},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={2167--2175},
    year={2016}}

    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    dataset_dir = 'vehicleid'

    def __init__(self, root='datasets', verbose=True, test_size=2400, **kwargs):
        super(VehicleID, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.test_size = test_size

        if self.test_size == 800:
            self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery_800')
            self.test_list = osp.join(self.split_dir, 'test_list_800.txt')
        elif self.test_size == 1600:
            self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery_1600')
            self.test_list = osp.join(self.split_dir, 'test_list_1600.txt')
        elif self.test_size == 2400:
            self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery_2400')
            self.test_list = osp.join(self.split_dir, 'test_list_2400.txt')

        print(self.gallery_dir)

        self.check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)
        query, gallery = self.process_split(relabel=True)

        self.train = train
        self.query = query
        self.gallery = gallery

        if verbose:
            print('=> VehicleID loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if self.test_size not in [800, 1600, 2400]:
            raise RuntimeError('"{}" is not available'.format(self.test_size))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))

    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 1 # don't have camid information use 1 for all
            img_path = osp.join(self.img_dir, name+'.jpg')
            output.append((img_path, pid, camid))
        return output

    def process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
        #pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid = int(re.search(r'([-\d]+)', img_path).group())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(re.search(r'([-\d]+)', img_path).group())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 131640  # pid == 0 means background
            pid = pid2label[pid]
            camid = 1
            dataset.append((img_path, pid, camid))

        return dataset

    def process_split(self, relabel=False):
        
        test_pid_dict = defaultdict(list)
        with open(self.test_list) as f_test:
            test_data = f_test.readlines()
            for data in test_data:
                name, pid = data.split(' ')
                test_pid_dict[pid].append([name, pid])
        test_pids = list(test_pid_dict.keys())
        num_test_pids = len(test_pids)
        assert num_test_pids == self.test_size, 'There should be {} vehicles for testing,' \
                                                ' but but got {}, please check the data'\
                                                .format(self.test_size, num_test_pids)

        query_data = []
        gallery_data = []

        # for each test id, random choose one image for gallery
        # and the other ones for query.
        for pid in test_pids:
            imginfo = test_pid_dict[pid]
            sample = random.choice(imginfo)
            imginfo.remove(sample)
            gallery_data.extend(imginfo)
            query_data.append(sample)

        query = self.parse_img_pids(query_data)
        gallery = self.parse_img_pids(gallery_data)
        return  query, gallery

