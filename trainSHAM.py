from config import args
from data_loader import load_labels
import numpy as np
import utils

print('SHAM starting...')
train_labels = load_labels(args.datasets)
num_classes = train_labels.shape[1]
inx = np.arange(train_labels.shape[0])
np.random.shuffle(inx)
available_labels = train_labels[inx[0: args.available_num]] if args.available_num > 0 else []
args.W = utils.SHAM(num_classes, args.output_shape, args.gama, available_labels=available_labels)
print('SHAM finished.')