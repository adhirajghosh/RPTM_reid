from __future__ import print_function
from __future__ import division
import os
import os.path as osp
import time
import torch
import numpy as np
import numpy.ma as ma
import random
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

from eval import do_test
from utils.loggers import RankLogger
from utils.torchtools import accuracy, save_checkpoint
from utils.functions import search, strint
from utils.avgmeter import AverageMeter
from utils.visualtools import visualize_ranked_results


def do_train(cfg, trainloader, train_dict, data_tfr, testloader_dict, dm,
             model, optimizer, scheduler, criterion_htri,criterion_xent):
    ranklogger = RankLogger(cfg.DATASET.SOURCE_NAME, cfg.DATASET.TARGET_NAME)
    gms = train_dict['gms']
    pidx = train_dict['pidx']
    folders = []
    for fld in os.listdir(cfg.DATASET.SPLIT_DIR):
        folders.append(fld)
    # data_index = search_index(gms, cfg.DATASET.SPLIT_DIR, folders)
    data_index = search(cfg.DATASET.SPLIT_DIR)

    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        losses = AverageMeter()
        xent_losses = AverageMeter()
        htri_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()

        model.train()
        for p in model.parameters():
            p.requires_grad = True  # open all layers

        end = time.time()
        for batch_idx, (img, label, index, pid, _) in enumerate(trainloader):

            trainX, trainY = torch.zeros((cfg.SOLVER.TRAIN_BATCH_SIZE * 3, 3, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH), dtype=torch.float32), torch.zeros(
                (cfg.SOLVER.TRAIN_BATCH_SIZE * 3), dtype=torch.int64)

            for i in range(cfg.SOLVER.TRAIN_BATCH_SIZE):

                labelx = str(label[i])
                # print(labelx)
                indexx = int(index[i])
                cidx = int(pid[i])
                if indexx > len(gms[labelx]) - 1:
                    indexx = len(gms[labelx]) - 1
                a = gms[labelx][indexx]

                if cfg.MODEL.RPTM_SELECT == 'min':
                    threshold = np.arange(10)
                elif cfg.MODEL.RPTM_SELECT == 'mean':
                    threshold = np.arange(np.amax(gms[labelx][indexx])//2)
                elif cfg.MODEL.RPTM_SELECT == 'max':
                    threshold = np.arange(np.amax(gms[labelx][indexx]))
                else:
                    threshold = np.arange(np.amax(gms[labelx][indexx]) // 2) #defaults to mean

                minpos = np.argmin(ma.masked_where(a == threshold, a))
                pos_dic = data_tfr[data_index[cidx][1] + minpos]
                # print(pos_dic[1])
                neg_label = int(labelx)
                while True:
                    neg_label = random.choice(range(1, 770))
                    if neg_label is not int(labelx) and os.path.isdir(
                            os.path.join(cfg.DATASET.SPLIT_DIR, strint(neg_label, 'veri'))) is True:
                        break
                negative_label = strint(neg_label, 'veri')
                neg_cid = pidx[negative_label]
                neg_index = random.choice(range(0, len(gms[negative_label])))

                neg_dic = data_tfr[data_index[neg_cid][1] + neg_index]
                trainX[i] = img[i]
                trainX[i + cfg.SOLVER.TRAIN_BATCH_SIZE] = pos_dic[0]
                trainX[i + (cfg.SOLVER.TRAIN_BATCH_SIZE * 2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i + cfg.SOLVER.TRAIN_BATCH_SIZE] = pos_dic[3]
                trainY[i + (cfg.SOLVER.TRAIN_BATCH_SIZE * 2)] = neg_dic[3]
            optimizer.zero_grad()
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            outputs, features = model(trainX)
            xent_loss = criterion_xent(outputs[0:cfg.SOLVER.TRAIN_BATCH_SIZE], trainY[0:cfg.SOLVER.TRAIN_BATCH_SIZE])
            htri_loss = criterion_htri(features, trainY)

            
            loss = cfg.LOSS.LAMBDA_HTRI * htri_loss + cfg.LOSS.LAMBDA_XENT * xent_loss

            if cfg.SOLVER.USE_AMP:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            for param_group in optimizer.param_groups:
                # print(param_group['lr'] )
                lrrr = str(param_group['lr'])

            batch_time.update(time.time() - end)
            losses.update(loss.item(), trainY.size(0))
            htri_losses.update(htri_loss.item(), trainY.size(0))
            accs.update(accuracy(outputs[0:cfg.SOLVER.TRAIN_BATCH_SIZE], trainY[0:cfg.SOLVER.TRAIN_BATCH_SIZE])[0])

            if (batch_idx) % cfg.MISC.PRINT_FREQ == 0:
                print('Train ', end=" ")
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'lr {lrrr} \t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    loss=losses,
                    acc=accs,
                    lrrr=lrrr,
                ))

            end = time.time()

        scheduler.step()
        print('=> Test')

        for name in cfg.DATASET.TARGET_NAME:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            rank1, distmat, rank2, distmat_re = do_test(model, queryloader, galleryloader, cfg.TEST.TEST_BATCH_SIZE, cfg.MISC.USE_GPU, cfg.DATASET.TARGET_NAME[0])
            
            ranklogger.write(name, epoch + 1, rank1)
            ranklogger.write(name, epoch + 1, rank2)
            
            if (epoch + 1) == cfg.SOLVER.MAX_EPOCHS and cfg.TEST.VIS_RANK == True:
                visualize_ranked_results(
                    distmat_re, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(cfg.MISC.SAVE_DIR, 'ranked_results', name),
                    topk=20)

        del queryloader
        del galleryloader
        del distmat
        # print(torch.cuda.memory_allocated(),torch.cuda.memory_cached())
        torch.cuda.empty_cache()

        if (epoch + 1) == cfg.SOLVER.MAX_EPOCHS:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'rank1': rank2,
                'epoch': epoch + 1,
                'arch': cfg.MODEL.ARCH,
                'optimizer': optimizer.state_dict(),
            }, cfg.MISC.SAVE_DIR, cfg.SOLVER.OPTIMIZER_NAME)