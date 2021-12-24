import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--seed', type=int, default=0)

# misc
parser.add_argument('--prefix', type=str, default='checkpoint')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--multiprocessing', action='store_true', default=False)
parser.add_argument('--view', type=int, default=-1)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lr_update', type=int, default=100)

parser.add_argument('--batch_sizes', type=list, default=[64, 64, 64, 32, 64]) ## 64
parser.add_argument('--lr', type=list, default=[1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
parser.add_argument('--output_shape', type=int, default=128)
parser.add_argument('--alpha', type=float, default=0.02) # nus_wide_tc21 16:0.02   32: 0.001   64: 0.003   128: 0.003
parser.add_argument('--gama', type=float, default=1.)
parser.add_argument('--datasets', type=str, default='mirflickr25k') # mirflickr25k nus_wide_tc21 MSCOCO-1, MSCOCO_doc2vec xmedia nus_wide_tc10 IAPR-TC12 wiki_doc2vec MSCOCO_bert xmedianet_full
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--available_num', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)

args = parser.parse_args()
