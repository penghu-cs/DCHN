import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import scipy
import scipy.io as sio


def main(args):
    from DCHN import Solver
    solver = Solver(args)
    cudnn.benchmark = True
    if args.mode == 'train':
        ret = solver.train()
    elif args.mode == 'eval':
        ret =  solver.eval()
    print(args)
    return ret

if __name__ == '__main__':
    from config import args
    import numpy as np
    from torch.backends import cudnn
    cudnn.enabled = False
    results = main(args)
