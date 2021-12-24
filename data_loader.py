import scipy.io as sio
import h5py
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np

class NDataset(Dataset):
    def __init__(self, data, inx, labels=None, transform=None, is_path=False, root='./'):

        self.data = data
        self.labels = labels
        self.inx = inx
        self.transform = transform
        self.is_path = is_path
        self.root = root

    def __getitem__(self, index):
        inx = self.inx[index]
        if self.is_path:
            if self.data[inx][-4::] == '.txt':
                pass
            else:
                img = Image.open(os.path.join(self.root, self.data[inx]))
                if img.mode == 'L':
                    img = img.convert('RGB')
                return img if self.transform is None else self.transform(img), self.labels[inx] if self.labels is not None else -1
        else:
            return self.data[inx] if self.transform is None else self.transform(self.data[inx]), self.labels[inx] if self.labels is not None else -1

    def __len__(self):
        # print(self.data.shape[0] if type(self.data) is np.ndarray else len(self.data))
        return self.inx.shape[0] if type(self.inx) is np.ndarray else len(self.inx)

def load_data(data_name, view):
    if 'xmedia' in data_name.lower():
        data, labels, train_inx, query_inx, retrieval_inx = loadXMedia()
        if view >= 0:
            data, labels, train_inx, query_inx, retrieval_inx = [data[view]], [labels[view]], [train_inx[view]], [query_inx[view]], [retrieval_inx[view]]
    elif view == 0:
        data, labels, train_inx, query_inx, retrieval_inx = load_img(data_name)
        data, labels, train_inx, query_inx, retrieval_inx = [data], [labels], [train_inx], [query_inx], [retrieval_inx]
    elif view == 1:
        data, labels, train_inx, query_inx, retrieval_inx = load_txt(data_name)
        data, labels, train_inx, query_inx, retrieval_inx = [data], [labels], [train_inx], [query_inx], [retrieval_inx]
    elif view == -1:
        img = load_img(data_name)
        txt = load_txt(data_name)
        data, labels, train_inx, query_inx, retrieval_inx = [img[0], txt[0]], [img[1], txt[1]], [img[2], txt[2]], [img[3], txt[3]], [img[4], txt[4]]
    else:
        # TODO
        raise RuntimeError('Undefined view!')

    return data, labels, train_inx, query_inx, retrieval_inx


def load_labels(data_name):
    if data_name == 'nus_wide_tc21':
        root = '../../DeepMDA/datasets/NUS-WIDE-TC21/'
        train_size = 10500
        query_size = 2100
        labels = sio.loadmat(root + 'nus-wide-tc21-lall-clean.mat')['LAll']
        train_labels = labels[query_size: query_size + train_size]
    elif data_name == 'mirflickr25k':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        labels = sio.loadmat(root + 'mirflickr25k-lall-rand.mat')['LAll']
        train_size = 10000
        query_size = 2000
        train_labels = labels[query_size: query_size + train_size]
    elif data_name == 'mirflickr25k_raw':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        train_size = 10000
        query_size = 2000
        labels = sio.loadmat(root + 'mirflickr25k-lall.mat')['LAll']
        train_labels = labels[query_size: query_size + train_size]
    elif data_name == 'IAPR-TC12':
        train_size = 10000
        file_path = '../../DeepMDA/datasets/IAPR-TC12/iapr-tc12.mat'
        data = sio.loadmat(file_path)
        train_labels = data['databaseL'][0: train_size]
    elif data_name == 'MSCOCO_doc2vec':
        path = '../../DeepMDA/datasets/MSCOCO/MSCOCO_deep_doc2vec_data.h5py'
        data = h5py.File(path)
        labels = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
        train_size = 10000
        query_size = 5000
        train_labels = labels[query_size: query_size + train_size]
    elif data_name == 'XMedia':
        data_img, labels, train_inx, query_inx, retrieval_inx = loadXMedia()
        train_labels = labels[0][train_inx[0]]
    else:
        raise RuntimeError('Unsupported dataset!')

    return train_labels


def load_img(data_name):
    import numpy as np
    if data_name == 'nus_wide_tc21':
        root = '../../DeepMDA/datasets/NUS-WIDE-TC21/'
        train_size = 10500
        query_size = 2100
        data_img = sio.loadmat(root + 'nus-wide-tc21-xall-vgg-clean.mat')['XAll'].astype('float32')
        labels = sio.loadmat(root + 'nus-wide-tc21-lall-clean.mat')['LAll']

    elif data_name == 'nus_wide_tc10':
        root = '../../DeepMDA/datasets/NUS-WIDE-TC10/'
        # inx = sio.loadmat(root + 'nus-wide-tc21-param.mat')['param'][()]
        train_size = 10500
        query_size = 2100
        data_img = sio.loadmat(root + 'nus-wide-tc10-xall-vgg.mat')['XAll'].astype('float32')
        labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']

    elif data_name == 'mirflickr25k':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        data_img = sio.loadmat(root + 'mirflickr25k-iall-vgg-rand.mat')['XAll'].astype('float32')
        labels = sio.loadmat(root + 'mirflickr25k-lall-rand.mat')['LAll']
        train_size = 10000
        query_size = 2000

    elif data_name == 'mirflickr25k_raw':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        train_size = 10000
        query_size = 2000
        data_img = h5py.File(root + 'mirflickr25k-iall.mat', 'r')['IAll']
        labels = sio.loadmat(root + 'mirflickr25k-lall.mat')['LAll']

    elif data_name == 'IAPR-TC12':
        train_size = 10000
        file_path = '../../DeepMDA/datasets/IAPR-TC12/iapr-tc12.mat'
        data = sio.loadmat(file_path)
        retrieval_img = data['VDatabase'].astype('float32')
        retrieval_labels = data['databaseL']
        query_img = data['VTest'].astype('float32')
        query_labels = data['testL']
        query_size = query_labels.shape[0]
        data_img = np.concatenate([query_img, retrieval_img])
        labels = np.concatenate([query_labels, retrieval_labels])

    elif data_name == 'MSCOCO_doc2vec':
        path = '../../DeepMDA/datasets/MSCOCO/MSCOCO_deep_doc2vec_data.h5py'
        data = h5py.File(path)
        data_img = np.concatenate([data['train_imgs_deep'][()], data['test_imgs_deep'][()]], axis=0)
        labels = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
        train_size = 10000
        query_size = 5000

    inx = np.arange(labels.shape[0])
    train_inx = inx[query_size: query_size + train_size]
    query_inx = inx[0: query_size]
    retrieval_inx = inx[query_size::]
    return data_img, labels, train_inx, query_inx, retrieval_inx

def load_txt(data_name):
    import numpy as np
    if data_name == 'nus_wide_tc21':
        root = '../../DeepMDA/datasets/NUS-WIDE-TC21/'
        train_size = 10500
        query_size = 2100
        data_txt = sio.loadmat(root + 'nus-wide-tc21-yall-clean.mat')['YAll'].astype('float32')
        labels = sio.loadmat(root + 'nus-wide-tc21-lall-clean.mat')['LAll']

    elif data_name == 'nus_wide_tc10':
        root = '../../DeepMDA/datasets/NUS-WIDE-TC10/'
        # inx = sio.loadmat(root + 'nus-wide-tc21-param.mat')['param'][()]
        train_size = 10500
        query_size = 2100
        data_txt = sio.loadmat(root + 'nus-wide-tc10-yall.mat')['YAll'][()].T.astype('float32')
        labels = sio.loadmat(root + 'nus-wide-tc10-lall.mat')['LAll']

    elif data_name == 'mirflickr25k':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        data_txt = sio.loadmat(root + 'mirflickr25k-yall-rand.mat')['YAll'].astype('float32')
        labels = sio.loadmat(root + 'mirflickr25k-lall-rand.mat')['LAll']
        train_size = 10000
        query_size = 2000

    elif data_name == 'mirflickr25k_raw':
        root = '../../DeepMDA/datasets/MIRFLICKR25K/'
        train_size = 10000
        query_size = 2000
        data_txt = sio.loadmat(root + 'mirflickr25k-yall.mat')['YAll'].astype('float32')
        labels = sio.loadmat(root + 'mirflickr25k-lall.mat')['LAll']

    elif data_name == 'IAPR-TC12':
        train_size = 10000
        file_path = '../../DeepMDA/datasets/IAPR-TC12/iapr-tc12.mat'
        data = sio.loadmat(file_path)
        retrieval_txt = data['YDatabase'].astype('float32')
        retrieval_labels = data['databaseL']
        query_txt = data['YTest'].astype('float32').astype('float32')
        query_labels = data['testL']
        query_size = query_labels.shape[0]
        data_txt = np.concatenate([query_txt, retrieval_txt])
        labels = np.concatenate([query_labels, retrieval_labels])

    elif data_name == 'MSCOCO_doc2vec':
        path = '../../DeepMDA/datasets/MSCOCO/MSCOCO_deep_doc2vec_data.h5py'
        data = h5py.File(path)
        data_txt = np.concatenate([data['train_text'][()], data['test_text'][()]], axis=0).astype('float32')
        labels = np.concatenate([data['train_imgs_labels'][()], data['test_imgs_labels'][()]], axis=0)
        train_size = 10000
        query_size = 5000
    inx = np.arange(labels.shape[0])
    train_inx = inx[query_size: query_size + train_size]
    query_inx = inx[0: query_size]
    retrieval_inx = inx[query_size::]
    return data_txt, labels, train_inx, query_inx, retrieval_inx


def loadXMedia():
    path = '../../DeepMDA/datasets/XMedia&Code/XMediaFeatures.mat'
    # path = '../datasets/XMedia&Code/XMediaFeaturesSplit.mat'
    MAP = -1
    req_rec, b_wv_matrix = False, False
    all_data = sio.loadmat(path)
    A_te = all_data['A_te'].astype('float32')  # Features of test set for audio data, MFCC feature
    A_tr = all_data['A_tr'].astype('float32')  # Features of training set for audio data, MFCC feature
    d3_te = all_data['d3_te'].astype('float32')  # Features of test set for 3D data, LightField feature
    d3_tr = all_data['d3_tr'].astype('float32')  # Features of training set for 3D data, LightField feature
    I_te_CNN = all_data['I_te_CNN'].astype('float32')  # Features of test set for image data, CNN feature
    I_tr_CNN = all_data['I_tr_CNN'].astype('float32')  # Features of training set for image data, CNN feature
    T_te_BOW = all_data['T_te_BOW'].astype('float32')  # Features of test set for text data, BOW feature
    T_tr_BOW = all_data['T_tr_BOW'].astype('float32')  # Features of training set for text data, BOW feature
    V_te_CNN = all_data['V_te_CNN'].astype('float32')  # Features of test set for video(frame) data, CNN feature
    V_tr_CNN = all_data['V_tr_CNN'].astype('float32')  # Features of training set for video(frame) data, CNN feature
    te3dCat = all_data['te3dCat'].reshape([-1]).astype('int64')  # category label of test set for 3D data
    tr3dCat = all_data['tr3dCat'].reshape([-1]).astype('int64')  # category label of training set for 3D data
    teAudCat = all_data['teAudCat'].reshape([-1]).astype('int64')  # category label of test set for audio data
    trAudCat = all_data['trAudCat'].reshape([-1]).astype('int64')  # category label of training set for audio data
    teImgCat = all_data['teImgCat'].reshape([-1]).astype('int64')  # category label of test set for image data
    trImgCat = all_data['trImgCat'].reshape([-1]).astype('int64')  # category label of training set for image data
    teVidCat = all_data['teVidCat'].reshape([-1]).astype('int64')  # category label of test set for video(frame) data
    trVidCat = all_data['trVidCat'].reshape([-1]).astype(
        'int64')  # category label of training set for video(frame) data
    teTxtCat = all_data['teTxtCat'].reshape([-1]).astype('int64')  # category label of test set for text data
    trTxtCat = all_data['trTxtCat'].reshape([-1]).astype('int64')  # category label of training set for text data

    train_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
    test_data = [I_te_CNN, T_te_BOW, A_te, d3_te, V_te_CNN]
    valid_data = [I_tr_CNN, T_tr_BOW, A_tr, d3_tr, V_tr_CNN]
    train_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]
    query_labels = [teImgCat, teTxtCat, teAudCat, te3dCat, teVidCat]
    valid_labels = [trImgCat, trTxtCat, trAudCat, tr3dCat, trVidCat]

    classes = np.unique(trImgCat).reshape([1, -1])
    for v in range(len(train_labels)):
        train_labels[v] = (train_labels[v].reshape([-1, 1]) == classes).astype('float32')
        query_labels[v] = (query_labels[v].reshape([-1, 1]) == classes).astype('float32')
        valid_labels[v] = (valid_labels[v].reshape([-1, 1]) == classes).astype('float32')

    data, labels, train_inx, query_inx, retrieval_inx = [], [], [], [], []
    for v in range(len(train_labels)):
        data.append(np.concatenate([test_data[v], valid_data[v]]))
        labels.append(np.concatenate([query_labels[v], valid_labels[v]]))
        train_size = train_data[v].shape[0]
        query_size = test_data[v].shape[0]

        inx = np.arange(labels[v].shape[0])
        tr_inx = inx[query_size: query_size + train_size]
        te_inx = inx[0: query_size]
        re_inx = inx[query_size::]
        train_inx.append(tr_inx)
        query_inx.append(te_inx)
        retrieval_inx.append(re_inx)
    return data, labels, train_inx, query_inx, retrieval_inx
