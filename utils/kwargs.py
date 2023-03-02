from __future__ import print_function
from __future__ import division

def return_kwargs(cfg):
    if cfg.DATASET.SOURCE_NAME[0] == 'vehicleid':
        dataset_kwargs = {
            'source_names': cfg.DATASET.SOURCE_NAME,
            'target_names': cfg.DATASET.TARGET_NAME,
            'root': cfg.DATASET.ROOT_DIR,
            'height': cfg.INPUT.HEIGHT,
            'width': cfg.INPUT.WIDTH,
            'test_size': cfg.TEST.TEST_SIZE,
            'train_batch_size': cfg.SOLVER.TRAIN_BATCH_SIZE,
            'test_batch_size': cfg.TEST.TEST_BATCH_SIZE,
            'train_sampler': cfg.DATALOADER.SAMPLER,
            'random_erase': cfg.INPUT.RANDOM_ERASE,
            'color_jitter': cfg.INPUT.JITTER,
            'color_aug': cfg.INPUT.AUG
        }
    else:
        dataset_kwargs = {
            'source_names': cfg.DATASET.SOURCE_NAME,
            'target_names': cfg.DATASET.TARGET_NAME,
            'root': cfg.DATASET.ROOT_DIR,
            'height': cfg.INPUT.HEIGHT,
            'width': cfg.INPUT.WIDTH,
            'test_size': cfg.TEST.TEST_SIZE,
            'train_batch_size': cfg.SOLVER.TRAIN_BATCH_SIZE,
            'test_batch_size': cfg.TEST.TEST_BATCH_SIZE,
            'train_sampler': cfg.DATALOADER.SAMPLER,
            'random_erase': cfg.INPUT.RANDOM_ERASE,
            'color_jitter': cfg.INPUT.JITTER,
            'color_aug': cfg.INPUT.AUG
        }

    transform_kwargs = {
        'height': cfg.INPUT.HEIGHT,
        'width': cfg.INPUT.WIDTH,
        'random_erase': cfg.INPUT.RANDOM_ERASE,
        'color_jitter': cfg.INPUT.JITTER,
        'color_aug': cfg.INPUT.AUG
    }

    optimizer_kwargs = {
        'optim': cfg.SOLVER.OPTIMIZER_NAME,
        'lr': cfg.SOLVER.BASE_LR,
        'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
        'momentum': cfg.SOLVER.MOMENTUM,
        'sgd_dampening': cfg.SOLVER.SGD_DAMP,
        'sgd_nesterov': cfg.SOLVER.NESTEROV
    }

    lr_scheduler_kwargs = {
        'lr_scheduler': cfg.SOLVER.LR_SCHEDULER,
        'stepsize': cfg.SOLVER.STEPSIZE,
        'gamma': cfg.SOLVER.GAMMA
    }

    return dataset_kwargs, transform_kwargs, optimizer_kwargs, lr_scheduler_kwargs