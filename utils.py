import torch
import numpy as np
import sys
import scipy.spatial
from scipy.linalg import solve_sylvester, svd
import scipy.io as sio
import os

def getSHAM(num_classes, output_shape, gama, available_num):
    file_name = 'SHAM/SHAM_W_C%d_O%d_G%g_A%d.mat' % (num_classes, output_shape, gama, available_num)
    if os.path.exists(file_name):
        W = sio.loadmat(file_name)['W']
    else:
        raise RuntimeError('Failed to open SHAM')
    return W

def SHAM(num_classes, output_shape, gama, available_labels=[], save_loss=False):
    file_name = 'SHAM/SHAM_W_C%d_O%d_G%g_A%d.mat' % (num_classes, output_shape, gama, len(available_labels))
    import os
    if os.path.exists(file_name):
        W = sio.loadmat(file_name)['W']
        print('SHAM has been trained. W has been reused!')
    else:
        import numpy as np
        np.random.seed(5555)
        W = np.random.randn(num_classes, output_shape)
        U, S, V = svd(W)
        S = np.eye(W.shape[0], W.shape[1])
        W = np.dot(U, np.dot(S, V))
        hash_opt = lambda x: (x > 0.).astype('float32') * 2 - 1

        inx = np.arange(num_classes)
        np.random.shuffle(inx)
        rand_labels = np.identity(num_classes)[:, inx]
        if len(available_labels) > 0:
            rand_labels = np.concatenate([rand_labels, available_labels])

        cmp_loss = lambda x, b, w: np.abs((x - np.dot(b, w.T))).sum(1).mean() + gama * np.abs((b - np.dot(x, w))).sum(1).mean()
        B = hash_opt((1 + gama) * np.dot(rand_labels, np.dot(W, np.linalg.inv(
            np.dot(W.T, W) + gama * np.identity(W.shape[1])).T)))
        loss = cmp_loss(rand_labels, B, W)
        print('iters %d: %.5f' % (0, loss))
        loss_dict = {}
        losses = []
        losses.append(loss)
        eye = np.identity(W.shape[1])
        for iter in range(50):
            B = hash_opt(
                (1 + gama) * np.dot(rand_labels, np.dot(W, np.linalg.inv(np.dot(W.T, W) + gama * eye).T)))
            A = np.dot(B.T, B)
            D = gama * np.dot(rand_labels.T, rand_labels)
            C = (1 + gama) * np.dot(B.T, rand_labels)
            W = solve_sylvester(A, D, C).T
            loss = cmp_loss(rand_labels, B, W)
            loss_dict[loss] = W
            losses.append(loss)
            if (iter + 1) % 20 == 0:
                print('iters %d: %.5f' % (iter + 1, loss))
        min_loss = min(list(loss_dict.keys()))
        W = loss_dict[min_loss]
        if save_loss:
            sio.savemat('SHAM_loss_C%d_O%d_G%g_A%d.mat' % (num_classes, output_shape, gama, len(available_labels)), {'loss': np.array(losses)})
        sio.savemat(file_name, {'W': W})
    return W

def save_checkpoint(state, filename='checkpoint.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    path = os.path.join(prefix, filename)
    while tries:
        try:
            torch.save(state, path)
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error

def to_tensor(x, cuda_id=0):
    x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.cuda(cuda_id)
    return x

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.numpy()

import scipy
def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    # numcases = dist.shape[1]
    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1] # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum) # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, query_L, retrieval_L):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map

# def predict(model, data, batch_size=32, cuda_id=0, isLong=False):
#     batch_count = int(np.ceil(data.shape[0] / float(batch_size)))
#     results = []
#     with torch.no_grad():
#         for i in range(batch_count):
#             batch = to_tensor(data[i * batch_size: (i + 1) * batch_size], cuda_id)
#             batch = batch.long() if isLong else batch
#             results.append(to_data(model(batch)))
#             # results.append(to_data(model(batch) > 0.5))
#     return np.concatenate(results)

def predict(model, dataloader, cuda_id=0):
    results, labels = [], []
    with torch.no_grad():
        for _, (d, t) in enumerate(dataloader):
            batch = to_tensor(d, cuda_id)
            results.append(to_data(model(batch)))
            labels.append(t)
    return np.concatenate(results), np.concatenate(labels)

def show_progressbar(rate, *args, **kwargs):
    '''
    :param rate: [current, total]
    :param args: other show
    '''
    inx = rate[0] + 1
    count = rate[1]
    bar_length = 30
    rate[0] = int(np.around(rate[0] * float(bar_length) / rate[1])) if rate[1] > bar_length else rate[0]
    rate[1] = bar_length if rate[1] > bar_length else rate[1]
    num = len(str(count))
    str_show = ('\r%' + str(num) + 'd / ' + '%' + str(num) + 'd  (%' + '3.2f%%) [') % (inx, count, float(inx) / count * 100)
    for i in range(rate[0]):
        str_show += '='

    if rate[0] < rate[1] - 1:
        str_show += '>'

    for i in range(rate[0], rate[1] - 1, 1):
        str_show += '.'
    str_show += '] '
    for l in args:
        str_show += ' ' + str(l)

    for key in kwargs:
        try:
            str_show += ' ' + key + ': %.4f' % kwargs[key]
        except Exception:
            str_show += ' ' + key + ': ' + str(kwargs[key])
    if inx == count:
        str_show += '\n'

    sys.stdout.write(str_show)
    sys.stdout.flush()
