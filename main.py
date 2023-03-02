import random
import torch
import numpy as np
import os
import os.path as osp
import argparse
import sys
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
from config import cfg
from datasets import data_loader
from model import ft_net_SE, init_model, init_optimizer
from loss import CrossEntropyLoss, TripletLoss
from train import do_train
from eval import do_test
from utils.kwargs import return_kwargs
from utils.loggers import Logger
from utils.torchtools import count_num_param, accuracy, load_pretrained_weights, save_checkpoint
from utils.visualtools import visualize_ranked_results
from utils.functions import create_split_dirs

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    parser = argparse.ArgumentParser(description="Relation Preserving Triplet Mining for Object Re-identification")
    parser.add_argument(
        "--config_file", default="configs/veri_r101.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    #Load the config file
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(1234)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.GPU_ID

    output_dir = cfg.MISC.SAVE_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_kwargs, transform_kwargs, optimizer_kwargs, lr_scheduler_kwargs = return_kwargs(cfg)

    if cfg.MISC.FP16:
        fp16 = True

    use_gpu = cfg.MISC.USE_GPU
    log_name = './log_test.txt' if cfg.TEST.EVAL else './log_train.txt'
    sys.stdout = Logger(osp.join(cfg.MISC.SAVE_DIR, log_name))

    if not os.path.exists(cfg.DATASET.SPLIT_DIR):
        create_split_dirs(cfg)

    print("Running for RPTM: ", cfg.MODEL.RPTM_SELECT)
    print('Currently using GPU ', cfg.MODEL.GPU_ID)
    print('Initializing image data manager')

    trainloader, train_dict, data_tfr, testloader_dict, dm = data_loader(cfg, dataset_kwargs, transform_kwargs)

    print('Initializing model: {}'.format(cfg.MODEL.ARCH))

    model = init_model(cfg.MODEL.ARCH, train_dict['class_names'], loss={'xent', 'htri'}, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if cfg.MODEL.PRETRAIN_PATH != '':
        print("weights loaded")
        load_pretrained_weights(model, cfg.MODEL.PRETRAIN_PATH)

    if use_gpu:
        model = model.cuda()
    optimizer = init_optimizer(model, **optimizer_kwargs)
    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.STEPSIZE, gamma=cfg.SOLVER.GAMMA)

    criterion_xent = CrossEntropyLoss(num_classes=train_dict['num_train_pids'], use_gpu=use_gpu, label_smooth=True)
    criterion_htri = TripletLoss(margin=cfg.LOSS.MARGIN)

    if cfg.TEST.EVAL:
        print('Evaluate only')

        for name in cfg.DATASET.TARGET_NAME:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            _, distmat, _, distmat_re = do_test(model, queryloader, galleryloader, cfg.TEST.TEST_BATCH_SIZE, use_gpu, cfg.DATASET.TARGET_NAME[0])

            if cfg.TEST.VIS_RANK:
                visualize_ranked_results(
                    distmat_re, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(cfg.MISC.SAVE_DIR, 'ranked_results', name),
                    topk=20
                )
        return

    print('=> Start training')

    do_train(cfg,
          trainloader,
          train_dict,
          data_tfr,
          testloader_dict,
          dm,
          model,
          optimizer,
          scheduler,
          criterion_htri,
          criterion_xent,
          )


if __name__ == '__main__':
    main()