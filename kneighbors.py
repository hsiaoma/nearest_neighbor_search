#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import sys, json, logging, faiss, time
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
import numpy as np
from datetime import datetime, timedelta
np.random.seed(42)

def readVec(filepath, readmap = {}):
    mids = []
    data = []
    with open(filepath, 'r') as infile:
        for line in infile.readlines():
            components = line.rstrip('\n').split(' ')
            if len(components) == 1:
                components = line.rstrip('\n').split('\t')
            mid = components[0]
            if (len(readmap) != 0 and mid not in readmap):
                    continue
            feat = np.array(components[1 : ], dtype = 'float32')
            norm = np.linalg.norm(feat)
            if norm != 0:
                feat /= norm
            else:
                continue
            mids.append(mid)
            data.append(feat)
    data = np.array(data)
    mids = np.array(mids)
    return mids, data

def findneighbors(mids, data, query, k):
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(data)
    D, I = index.search(query, k)
    return np.array(mids)[I[:, 1 : ]], D[:, 1 : ].astype(float)

def approxneighbors(mids, data, query, k):
    nprob = 100
    index = faiss.index_factory(data.shape[1], "IVF4096, Flat")
    faiss.ParameterSpace().set_index_parameter(index, 'nprobe', nprob)

    indices = np.random.choice(data.shape[0], np.min([100 * 4096, int(data.shape[0])]), replace = False)
    logging.info('training')
    index.train(data[indices, : ])
    logging.info('building index')
    index.add(data)
    logging.info('probing... ')
    t0 = time.time()
    D, I = index.search(query, k)
    logging.info('searching time = %.3f' %((time.time() - t0) / 60))
    return np.array(mids)[I[:, 1 : ]], D[:, 1 : ].astype(float)

if __name__ == '__main__':

    k = 20

    logging.info('Reading data')

    mids, data = readVec('../data/'  + sys.argv[1] + '/media_id_merged.out')
    logging.info ('Total # of medias = %d' %len(mids))
    indices = np.random.choice(data.shape[0], 10 ** 3, replace = False)
    valset = data[indices, : ]

    logging.info('Approximate search')
    t0 = time.time()
    approx, D  = approxneighbors(mids, data, data, k = k + 1)
    logging.info('Approximate search completed, %.2f' %((time.time() - t0) / 60))

    logging.info('Exhaustive search')
    t0 = time.time()
    label, _ = findneighbors(mids, data, valset, k = k + 1)
    logging.info('Exhaustive search completed, %.2f' %((time.time() - t0) / 60))

    recall_list = []
    for i in range(len(label)):
        s1 = set(label[i])
        s2 = set(approx[indices[i]])
        recalled = len(s1.intersection(s2)) / float(k)
        recall_list.append(recalled)
    logging.info('Average Recall = %.3f' %np.mean(recall_list))

    with open('../data/%s/neighbor_list.out'%(sys.argv[1]), 'w') as f:
        for i in range(len(mids)):
            nei = [(int(approx[i][j]), (2.0 - D[i][j]) * 0.5) for j in range(len(approx[i]))]
            line = mids[i] + '\t' + json.dumps(nei) + '\n'
            f.write(line)

