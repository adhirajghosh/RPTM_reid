
from __future__ import print_function
from __future__ import division
from utils.reranking import re_ranking,re_ranking_numpy
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import numpy.ma as ma
import warnings
import pickle
from skimage import io, transform
from PIL import Image
import os.path as osp
import random
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms, utils
from model import ft_net_SE
#from senet import ft_net_SE
#import models
from losses import triplet_loss, xent_loss
from loss import CrossEntropyLoss, TripletLoss
from utils.avgmeter import AverageMeter
from utils.iotools import check_isfile
from utils.loggers import Logger, RankLogger
from utils.torchtools import count_num_param, accuracy, load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from utils.visualtools import visualize_ranked_results
from utils.generaltools import set_random_seed
from optimizers import init_optimizer
from lr_schedulers import init_lr_scheduler
from functions import *
from datasets import init_imgreid_dataset
from data_loading import DukeDataset as dd
#from transform import Rescale, RandomCrop, ToTensor 
from transform import train_transforms
from test_loading import ImageDataManager
from evaluation import evaluate
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def test(model, queryloader, galleryloader, batch_size, use_gpu, ranks=[1, 5, 10], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    distmat_re = re_ranking(qf, gf, k1=80, k2=15, lambda_value=0.2)
    print('Computing CMC and mAP')
    # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, target_names)
    cmc_re, mAP_re = evaluate(distmat_re, q_pids, g_pids, q_camids, g_camids, 50)

    print('Re-Ranked Results ----------')
    print('mAP: {:.1%}'.format(mAP_re))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc_re[r - 1]))
    print('------------------')
    
    return cmc[0], distmat, cmc_re[0], distmat_re

def main():
    #GENERAL
    root_dir = "/home/jackie/datasets/"
    train_dir = '/home/jackie/datasets/duke/image_train/'
    split_dir = '/home/jackie/datasets/duke/train_split/'
    save_dir = './log/duke/'
    pkl_path = './gms/duke/'
    index_path = './pkl/duke/index.pkl'
    source = {'duke'}
    target = {'duke'}
    workers = 8
    height = 300
    width  = 150
    train_sampler = 'RandomSampler'

    #AUGMENTATION
    random_erase = True
    jitter = True
    aug = True

    #OPTIMIZATION
    opt = 'sgd'
    #lr = 0.003
    lr = 0.005
    weight_decay = 5e-4
    momentum = 0.9
    sgd_damp = 0.0
    nesterov = True
    #warmup_factor = 0.01
    #warmup_method = 'linear'

    #HYPERPARAMETER
    start = 0
    max_epoch = 80
    train_batch_size = 24
    test_batch_size = 100
    use_amp = True
    #SCHEDULER
    lr_scheduler = 'multi_step'
    stepsize = [20, 40, 60]
    gamma = 0.1

    #LOSS
    margin = 0.3
    #num_instances = 6
    lambda_tri = 1
    lambda_xent = 1

    #MODEL
    arch = 'SE_net'
    droprate = 0.2
    stride = 1 
    #erasing_p = 0.5
    pool = 'avg'

    #TEST SETTINGS
    #load_weights = './log/model_SE_net_sgd_18.pth.tar'
    load_weights = None

    #MISC
    use_gpu = True
    print_freq = 100
    gpu_id = 0
    vis_rank = True
    evaluate = False

    dataset_kwargs = {
        'source_names': source,
        'target_names': target,
        'root': root_dir,
        'height': height,
        'width': width,
        'train_batch_size': train_batch_size,
        'test_batch_size': test_batch_size,
        'train_sampler': train_sampler,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
        }
    transform_kwargs = {
        'height': height,
        'width': width,
        'random_erase': random_erase,
        'color_jitter': jitter,
        'color_aug': aug
    }

    optimizer_kwargs = {
        'optim': opt,
        'lr': lr,
        'weight_decay': weight_decay,
        'momentum': momentum,
        'sgd_dampening': sgd_damp,
        'sgd_nesterov': nesterov
        }

    
    fp16 = True
    use_gpu = torch.cuda.is_available()
    log_name = 'log_test.txt' if evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(save_dir, log_name))
    print("Running train_duke.py")
    print('Currently using GPU ', gpu_id)
    cudnn.benchmark = True

    print('Initializing image data manager')
    dataset = init_imgreid_dataset(root=root_dir, name='duke')
    num_train_pids = 0
    num_train_cams = 0
    train = []

    for img_path, pid, camid in dataset.train:
        
        path = img_path[-20:]
        #print(path)
        folder = path[0:4]
        pid += num_train_pids
        camid += num_train_cams
        train.append((path, folder, pid, camid))

    num_train_pids += dataset.num_train_pids
    class_names = num_train_pids
    num_train_cams += dataset.num_train_cams

    pid = 0
    pidx = {}
    for img_path, pid, camid in dataset.train:
        path = img_path[-20:]
        folder = path[0:4]
        pidx[folder] = pid
        pid+= 1

    
    pkl = {}
    entries = sorted(os.listdir(pkl_path))
    #print(entries)
    for name in entries:
        f = open((pkl_path+name), 'rb')
        if name=='featureMatrix.pkl':
            s = name[0:13]
        else:
            s = name[0:4]
        pkl[s] = pickle.load(f)
        f.close

    folders = []
    for fld in os.listdir(split_dir):
        folders.append(fld)
    #print(pkl)
    transform_t = train_transforms(**transform_kwargs)

    data_tfr = dd(pkl_file= index_path, dataset = train, root_dir=train_dir, transform=transform_t)
    trainloader = DataLoader(data_tfr, sampler=None,batch_size=train_batch_size, shuffle=True, num_workers=workers,pin_memory=False, drop_last=True)

    print('Initializing test data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs)
    testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(arch))
    #model = models.init_model(name=arch, num_classes=num_train_pids, loss={'xent', 'htri'}, last_stride = 2,pretrained=not no_pretrained, use_gpu=use_gpu)
    #model = ft_net_SE(class_names, layers, fc_dims = None, droprate = droprate, stride = stride, pool = pool, init_model = None, block = Bottleneck_IBN)

    model = ft_net_SE(class_names, droprate = droprate, stride = stride, pool = pool, init_model=None)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if load_weights is not None:
        print("weights loaded")
        load_pretrained_weights(model, load_weights)
    
    model = model.cuda()
    optimizer = init_optimizer(model, **optimizer_kwargs)
    #optimizer = init_optimizer(model)
    #model, optimizer = amp.initialize(model, optimizer, opt_level = "O1")
    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O2", 
        keep_batchnorm_fp32=True, loss_scale="dynamic")
        
    #scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=stepsize, gamma=gamma)

    criterion_xent = CrossEntropyLoss(num_classes=num_train_pids, use_gpu=use_gpu, label_smooth=True)
    criterion_htri = TripletLoss(margin=margin)

    if evaluate:
        print('Evaluate only')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            _, distmat,_, distmat_re = test(model, queryloader, galleryloader, train_batch_size, use_gpu, return_distmat=True)

            if vis_rank:
                visualize_ranked_results(
                    distmat_re, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(save_dir, 'ranked_results', name),
                    topk=20
                )
        return    

    ranklogger = RankLogger(source, target)
    print('=> Start training')

    data_index = search_index(pkl, split_dir, folders)
    #print(pkl)
    
    for epoch in range(start, max_epoch):
        losses = AverageMeter()
        #xent_losses = AverageMeter()
        htri_losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()

        model.train()
        for p in model.parameters():
            p.requires_grad = True    # open all layers

        end = time.time()
        for batch_idx, (img,label,index,pid, _) in enumerate(trainloader):
            #print (label)

            trainX, trainY = torch.zeros((train_batch_size*3,3,height, width), dtype=torch.float32), torch.zeros((train_batch_size*3), dtype = torch.int64)
            #pids = torch.zeros((batch_size*3), dtype = torch.int16)
            for i in range(train_batch_size):
 
                labelx = str(label[i])
                #print(labelx)
                indexx = int(index[i])
                cidx = int(pid[i])
                if indexx >len(pkl[labelx])-1:
                    indexx = len(pkl[labelx])-1
                a = pkl[labelx][indexx]
                threshold = np.arange(np.amax(pkl[labelx][indexx])//2)
                #threshold = np.arange(10)
                minpos = np.argmin(ma.masked_where(a==threshold, a)) 
                pos_dic = data_tfr[data_index[cidx][1]+minpos]
                #print(pos_dic[1])
                neg_label = int(labelx)
                while True:	
                    neg_label = random.choice(range(0, 7141))
                    n_l = strint(neg_label, 'duke')
                    if neg_label is not int(labelx) and os.path.isdir(os.path.join(split_dir, n_l)) is True:
                        break
                negative_label = n_l
                neg_cid = pidx[negative_label]
                neg_index = random.choice(range(0, len(pkl[negative_label])))

                neg_dic = data_tfr[data_index[neg_cid][1]+neg_index]
                trainX[i] = img[i]
                trainX[i+train_batch_size] = pos_dic[0]
                trainX[i+(train_batch_size*2)] = neg_dic[0]
                trainY[i] = cidx
                trainY[i+train_batch_size] = pos_dic[3]
                trainY[i+(train_batch_size*2)] = neg_dic[3]
            optimizer.zero_grad()
            trainX = trainX.cuda()
            trainY = trainY.cuda()
            outputs, features = model(trainX)
            xent_loss = criterion_xent(outputs[0:train_batch_size], trainY[0:train_batch_size])
            htri_loss = criterion_htri(features, trainY)

            #tri_loss = ranking_loss(features)
            #ent_loss = xent_loss(outputs[0:batch_size], trainY[0:batch_size], num_train_pids)
            
            loss = lambda_tri*htri_loss+lambda_xent*xent_loss
            
            if use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        
            #loss.backward()
            optimizer.step()
            for param_group in optimizer.param_groups:
                #print(param_group['lr'] )
                lrrr= str(param_group['lr'])

            batch_time.update(time.time() - end)
            losses.update(loss.item(), trainY.size(0))
            htri_losses.update(htri_loss.item(), trainY.size(0))
            accs.update(accuracy(outputs[0:train_batch_size], trainY[0:train_batch_size])[0])

            if (batch_idx) % print_freq == 0:
                print('Train ', end=" ")
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                    'lr {lrrr} \t'.format(
                    epoch + 1, batch_idx + 1, len(trainloader),
                    batch_time=batch_time,
                    loss = losses,
                    acc=accs,
                    lrrr=lrrr,
                ))

            end = time.time()
        
        scheduler.step()
        print('=> Test')

        for name in target:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            rank1, distmat, rank2, distmat_re = test(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank1)
            #rank2, distmat2 = test_rerank(model, queryloader, galleryloader, test_batch_size, use_gpu)
            ranklogger.write(name, epoch + 1, rank2)
            if (epoch+1) == max_epoch and vis_rank==True:
                visualize_ranked_results(
                    distmat_re, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(save_dir, 'ranked_results', name),
                    topk=20)

        del queryloader
        del galleryloader
        del distmat
        #print(torch.cuda.memory_allocated(),torch.cuda.memory_cached())
        torch.cuda.empty_cache()

        save_checkpoint({
            'state_dict': model.state_dict(),
            'rank1': rank2,
            'epoch': epoch + 1,
            'arch': arch,
            'optimizer': optimizer.state_dict(),
            }, save_dir, opt)
        

if __name__ == '__main__':
    main()
