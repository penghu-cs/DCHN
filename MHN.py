import torch
import numpy as np
import multiprocessing
import os
from torch import optim
import utils
from torch.autograd import Variable
from model import Dense_Net, ImgNet


class MHN(object):
    def __init__(self, config, train_dataloader, view):
        self.args = config
        self.output_shape = config.output_shape
        self.seed = config.seed
        # self.data, self.labels, self.train_inx, self.query_inx, self.retrieval_inx = view_data
        self.train_dataloader = train_dataloader
        self.input_shape = self.train_dataloader.dataset.data.shape[1]
        self.view = view
        self.num_classes = self.train_dataloader.dataset.labels.shape[1]

        if 'raw' in config.datasets:
            if self.view == 0:
                self.model = ImgNet(out_dim=self.output_shape)
            else:
                self.model = Dense_Net(input_dim=self.input_shape, out_dim=self.output_shape)
        else:
            self.model = Dense_Net(input_dim=self.input_shape, out_dim=self.output_shape)

        # train_dataset = data_loader.NDataset(self.data, self.train_inx, self.labels, transform=train_transform)
        # self.train_dataloader = data.DataLoader(train_dataset, batch_size=self.batch_sizes[self.view], shuffle=True, num_workers=num_workers, drop_last=False)

        # retrieval_dataset = data_loader.NDataset(self.data, self.retrieval_inx, self.labels, transform=test_transform)
        # self.retrieval_dataloader = data.DataLoader(retrieval_dataset, batch_size=self.batch_sizes[self.view], shuffle=False, num_workers=num_workers, drop_last=False)

        # query_dataset = data_loader.NDataset(self.data, self.query_inx, self.labels, transform=test_transform)
        # self.query_dataloader = data.DataLoader(query_dataset, batch_size=self.batch_sizes[self.view], shuffle=False, num_workers=num_workers, drop_last=False)

        self.lr = config.lr[view]
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_sizes = config.batch_sizes

        self.epochs = config.epochs
        self.available_num = config.available_num
        self.alpha = config.alpha
        self.gama = config.gama
        self.W = utils.getSHAM(self.num_classes, self.output_shape, self.gama, self.available_num)
        self.checkpoint_file = '{}_last_checkpoint_V{}_O{}_A{}.pth.tar'.format(self.args.datasets, self.view, self.output_shape, self.available_num)

    def to_var(self, x, cuda_id):
        """Converts numpy to variable."""
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if torch.cuda.is_available():
            x = x.cuda(cuda_id)
        return Variable(x)  # torch.autograd.Variable

    def to_data(self, x):
        """Converts variable to numpy."""
        try:
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy()
        except Exception as e:
            return x

    def to_hashing(self, y, W):
        if isinstance(y, torch.Tensor):
            if len(y.shape) == 1 or y.shape[1] == 1:
                tmp_ = (W[y] > 0).float() * 2 - 1
                train_y = tmp_
            else:
                train_y = ((y.float().mm(W) > 0.).float() * 2. - 1).detach()
                # train_y = y.float().mm(W).sign().detach()
            # train_y = (train_y / math.sqrt(train_y.shape[1])).detach()
            train_y.requires_grad = False
        else:
            if len(y.shape) == 1 or y.shape[1] == 1:
                tmp_ = (self.W[y] > 0) * 2 - 1
                train_y = tmp_
            else:
                train_y = (np.dot(y.reshape([-1, self.W.shape[0]]), self.W) > 0) * 2 - 1
        return train_y

    def criterion(self, x, y, labels, W):
        l2 = lambda _x, _y: ((_x - _y) ** 2).sum(1).mean()
        if isinstance(x, torch.Tensor):
            dist = x.mm(y.t()) / 2.
            sim = (labels.float().mm(labels.float().t()) > 0).float()
            loss1 = ((1. + dist.double().exp()).log() - (sim * dist).float()).sum(1).mean().float()
            loss2 = l2(x.mm(W.t()), labels)
            return self.alpha * loss1 + (1 - self.alpha) * loss2
        else:
            return (1 - self.alpha) * l2(x, y) + self.alpha * l2(np.dot(x, self.W.T), labels)

    def train_view(self, cuda_id):
        print('Start %d-th MHN!' % self.view)
        seed = self.seed
        import numpy as np
        np.random.seed(seed)
        import random as rn
        import torch
        rn.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import time


        start = time.time()
        if torch.cuda.is_available():
            self.model.cuda(cuda_id)

        model_params = []
        for name, params in self.model.named_parameters():
            if 'vgg' in name:
                model_params += [{'params': [params], 'lr': self.lr * 1e-1}]
                pass
            else:
                model_params += [{'params': [params]}]
        optimizer = optim.Adam(model_params, self.lr, [self.beta1, self.beta2])

        losses = []
        W = torch.tensor(self.W, requires_grad=False).cuda(cuda_id).float()
        criterion = lambda x, y, la: self.criterion(x, y, la, W)
        batch_count = len(self.train_dataloader)

        for epoch in range(self.epochs):
            print(('\nView ID: %d, Epoch %d/%d') % (self.view, epoch + 1, self.epochs))
            self.model.train()
            mean_loss = []
            # for batch_idx in range(batch_count):
            for batch_idx, (train_x, train_lab) in enumerate(self.train_dataloader):
                train_x = self.to_var(train_x, cuda_id)
                train_lab = self.to_var(train_lab, cuda_id)
                train_y = self.to_hashing(train_lab, W).float()

                optimizer.zero_grad()
                loss = criterion(self.model(train_x)[-1], train_y, train_lab)

                loss.backward()
                optimizer.step()
                mean_loss.append(self.to_data(loss))
                utils.show_progressbar([batch_idx, batch_count], loss=(loss.item() if batch_idx < batch_count - 1 else np.mean(mean_loss)))
            losses.append(np.mean(mean_loss))
            utils.save_checkpoint({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'opt': self.args,
                'loss': np.array(losses)
            }, filename=self.checkpoint_file, prefix=self.args.prefix)
            self.adjust_learning_rate(optimizer, epoch + 1)

        print('Training time: %.3f' % (time.time() - start))
        # query_pre = (utils.predict(lambda x: self.model(x)[-1].view([x.shape[0], -1]), self.query_dataloader, cuda_id=cuda_id).reshape([self.query_data[self.view].shape[0], -1]) > 0) * 2 - 1
        # retrieval_pre = (utils.predict(lambda x: self.model(x)[-1].view([x.shape[0], -1]), self.retrieval_dataloader, cuda_id=cuda_id).reshape([self.retrieval_data[self.view].shape[0], -1]) > 0) * 2 - 1

        return self.model

    def eval(self, eval_dataloader, cuda_id):
        self.model = self.model.cuda(cuda_id)
        self.model.eval()
        ret, lab = utils.predict(lambda x: self.model(x)[-1].view([x.shape[0], -1]), eval_dataloader, cuda_id=cuda_id)
        return (ret > 0) * 2 - 1, lab

    def adjust_learning_rate(self, optimizer, epoch):
        """
        Sets the learning rate to the initial LR
        decayed by 10 after opt.lr_update epoch
        """
        if (epoch % self.args.lr_update) == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    def load_checkpoint(self, checkpoint_file=None):
        checkpoint_file = os.path.join(self.args.prefix, self.checkpoint_file) if checkpoint_file is None else checkpoint_file
        ckp = torch.load(checkpoint_file)
        self.model.load_state_dict(ckp['model'])
        print('Load pretrained model at %d-th epoch.' % ckp['epoch'])
        print(ckp['opt'])
        return ckp['epoch'], ckp['model'], ckp['opt'], ckp['loss']

