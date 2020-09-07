from train import train
from initialization import parse_args
from test import test
import tensorflow as tf
tf.reset_default_graph()
args = parse_args()

if __name__ == '__main__':
    if args.name == 'train':
        train()
    if args.name == 'test':
        test()
