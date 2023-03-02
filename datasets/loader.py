import os
import pickle
from torch.utils.data import DataLoader
from .init_dataset import init_imgreid_dataset
from .transform import *
from .data_loading import VeriDataset as vd
from .data_loading import IdDataset as id
from .data_loading import DukeDataset as dd
from .test_loading import ImageDataManager

def data_loader(cfg, dataset_kwargs, transform_kwargs):
    dataset = init_imgreid_dataset(root=cfg.DATASET.ROOT_DIR, name=cfg.DATASET.SOURCE_NAME[0])
    num_train_pids = 0
    num_train_cams = 0
    train = []

    for img_path, pid, camid in dataset.train:
        # path = img_path[-24:]
        path = img_path.split('/', 4)[-1]
        if cfg.DATASET.SOURCE_NAME[0] == 'veri':
            folder = path.split('_', 1)[0][1:]
        else:
            folder = path.split('_', 1)[0]
        pid += num_train_pids
        camid += num_train_cams
        train.append((path, folder, pid, camid))

    num_train_pids += dataset.num_train_pids
    class_names = num_train_pids
    num_train_cams += dataset.num_train_cams

    pid = 0
    pidx = {}
    for img_path, pid, camid in dataset.train:
        path = img_path.split('/', 4)[-1]
        if cfg.DATASET.SOURCE_NAME[0] == 'veri':
            folder = path.split('_', 1)[0][1:]
        else:
            folder = path.split('_', 1)[0]
        pidx[folder] = pid
        pid += 1

    gms = {}
    entries = sorted(os.listdir(cfg.MISC.GMS_PATH))
    # print(entries)
    for name in entries:
        f = open((cfg.MISC.GMS_PATH + name), 'rb')
        if name == 'featureMatrix.pkl':
            s = name[0:13]
        else:
            s = name[0:3]
        gms[s] = pickle.load(f)
        f.close

    transform_t = train_transforms(**transform_kwargs)
    if cfg.DATASET.SOURCE_NAME[0] == 'veri':
        data_tfr = vd(pkl_file=cfg.MISC.INDEX_PATH, dataset=train, root_dir=cfg.DATASET.TRAIN_DIR, transform=transform_t)
    elif cfg.DATASET.SOURCE_NAME[0] == 'vehicleid':
        data_tfr = id(pkl_file=cfg.MISC.INDEX_PATH, dataset=train, root_dir=cfg.DATASET.TRAIN_DIR, transform=transform_t)
    elif cfg.DATASET.SOURCE_NAME[0] == 'duke':
        data_tfr = dd(pkl_file=cfg.MISC.INDEX_PATH, dataset=train, root_dir=cfg.DATASET.TRAIN_DIR, transform=transform_t)
    trainloader = DataLoader(data_tfr, sampler=None, batch_size=cfg.SOLVER.TRAIN_BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS,
                             pin_memory=False, drop_last=True)

    print('Initializing test data manager')
    dm = ImageDataManager(cfg.MISC.USE_GPU, **dataset_kwargs)
    testloader_dict = dm.return_dataloaders()
    train_dict = {}
    train_dict['class_names'] = class_names
    train_dict['num_train_pids'] = num_train_pids
    train_dict['gms'] = gms
    train_dict['pidx'] = pidx


    return trainloader, train_dict, data_tfr, testloader_dict, dm
