from __future__ import print_function
from __future__ import division
import time
import numpy as np
import torch
from utils.evaluation import evaluate, evaluate_vid
from utils.reranking import re_ranking
from utils.avgmeter import AverageMeter


def do_test(model, queryloader, galleryloader, batch_size, use_gpu, dataset, ranks=[1, 5, 10]):
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
    if dataset == 'vehicleid':
        cmc, mAP = evaluate_vid(distmat, q_pids, g_pids, q_camids, g_camids, 50)
    else:
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, 50)

    print('Results ----------')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    print('------------------')

    distmat_re = re_ranking(qf, gf, k1=80, k2=15, lambda_value=0.2)
    print('Computing CMC and mAP')
    if dataset == 'vehicleid':
        cmc_re, mAP_re = evaluate_vid(distmat_re, q_pids, g_pids, q_camids, g_camids, 50)
    else:
        cmc_re, mAP_re = evaluate(distmat_re, q_pids, g_pids, q_camids, g_camids, 50)
    print('Re-Ranked Results--')
    print('mAP: {:.1%}'.format(mAP_re))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc_re[r - 1]))
    print('------------------')

    return cmc[0], distmat, cmc_re[0], distmat_re