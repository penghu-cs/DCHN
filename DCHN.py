from random import normalvariate
import torch
import numpy as np
import multiprocessing
import time
from torch import optim
import utils
import data_loader
from torch.autograd import Variable
import scipy.io as sio
import torch.utils.data as data
import math
from MHN import MHN


class Solver(object):
    def __init__(self, config):
        wv_matrix = None
        self.args = config
        self.output_shape = config.output_shape
        self.datasets = config.datasets
        self.seed = config.seed
        self.multiprocessing = config.multiprocessing
        self.available_num = config.available_num
        self.view = config.view
        (self.data, self.labels, self.train_inx, self.test_inx, self.retrieval_inx, train_transforms, test_transforms) = data_loader.load_data(self.datasets, self.view)
        self.n_view = len(self.data)
        num_workers = self.args.num_workers
        self.models, self.train_dataloader, self.retrieval_dataloader, self.query_dataloader = [], [], [], []
        for v in range(self.n_view):
            view = v if self.view < 0 else self.view
            train_dataset = data_loader.NDataset(self.data[v], self.train_inx[v], self.labels[v], transform=train_transforms[v])
            self.train_dataloader.append(data.DataLoader(train_dataset, batch_size=config.batch_sizes[view], shuffle=True, num_workers=num_workers, drop_last=False))
            self.models.append(MHN(config, self.train_dataloader[v], view))

            retrieval_dataset = data_loader.NDataset(self.data[v], self.retrieval_inx[v], self.labels[v], transform=test_transforms[v])
            self.retrieval_dataloader.append(data.DataLoader(retrieval_dataset, batch_size=config.batch_sizes[view], shuffle=False, num_workers=num_workers, drop_last=False))

            test_dataset = data_loader.NDataset(self.data[v], self.test_inx[v], self.labels[v], transform=test_transforms[v])
            self.query_dataloader.append(data.DataLoader(test_dataset, batch_size=config.batch_sizes[view], shuffle=False, num_workers=num_workers, drop_last=False))

    def getDevice(self, v):
        return self.args.gpu_id if self.args.gpu_id >= 0 else (v + 1) % torch.cuda.device_count()

    def train(self):
        start = time.time()
        if self.multiprocessing and self.view < 0:
            # Old PyTorch Version <= 1.1.0
            # import torch.multiprocessing as mp
            # mp = mp.get_context('spawn')
            # self.resutls = mp.Manager().list(self.resutls)
            # process = []
            # start = time.time()
            # for v in range(self.n_view):
            #     process.append(mp.Process(target=self.train_view, args=(v,)))
            #     process[v].daemon = True
            # for v in range(self.n_view):
            #     process[v].start()
            # start = time.time()
            # for p in process:
            #     p.join()

            # New PyTorch Version >= 1.2.0
            import threading
            ths = []
            for v in range(self.n_view):
                cuda_id = self.getDevice(v)
                ths.append(threading.Thread(target=self.models[v].train_view, args=(cuda_id,)))
            for v in range(self.n_view):
                ths[v].start()
            for p in ths:
                p.join()

        elif self.view < 0:
            start = time.time()
            for v in range(self.n_view):
                cuda_id = self.getDevice(v)
                self.models[v].train_view(cuda_id)
        else:
            start = time.time()
            cuda_id = self.getDevice(self.view)
            self.models[0].train_view(cuda_id)
        end = time.time()
        runing_time = end - start
        print('The training time: ' + str(runing_time))

    def eval(self):
        for v in range(self.n_view):
            self.models[v].load_checkpoint()
            # print('View #%d: %d' % (v, self.retrieval_dataloader[v].dataset.data.shape[1]))

        retrieval = [self.models[v].eval(self.retrieval_dataloader[v], self.getDevice(v)) for v in range(self.n_view)]
        query = [self.models[v].eval(self.query_dataloader[v], self.getDevice(v)) for v in range(self.n_view)]
        results = np.zeros([self.n_view, self.n_view])
        for j in range(self.n_view):
            for i in range(self.n_view):
                if i == j:
                    continue
                results[j, i] = utils.fx_calc_map_multilabel_k(retrieval[i][0], retrieval[i][1], query[j][0], query[j][1], k=0, metric='hamming')
        # print("best_epoch: " + str(self.best_epoch) + ", \tresults: " + self.view_result(results))
        print("results: " + self.view_result(results))
        sio.savemat('features/' + self.datasets + '_test_' + str(self.available_num) + '_' + str(self.output_shape) + 'bit_features.mat', {'query': [query[v][0] for v in range(self.n_view)], 'query_labels': [query[v][1] for v in range(self.n_view)], 'retrieval': [retrieval[v][0] for v in range(self.n_view)], 'retrieval_labels': [retrieval[v][1] for v in range(self.n_view)]})
        return results

    def view_result(self, _acc):
        res = ''
        if type(_acc) is not list:
            res += ((' - mean: %.3f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail: ')
            for _i in range(self.n_view):
                for _j in range(self.n_view):
                    if _i != _j:
                        res += ('%.3f' % _acc[_i, _j]) + ' , '
        else:
            R = [50, 'ALL']
            for _k in range(len(_acc)):
                res += (' R = ' + str(R[_k]) + ': ')
                res += ((' - mean: %.3f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail: ')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.3f' % _acc[_k][_i, _j]) + ' , '
        return res