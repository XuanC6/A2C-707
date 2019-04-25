# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import sys, os, glob
import tensorflow as tf

# from trainer import Trainer
from trainer import Trainer
from config import Configuration

#gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
tf.enable_eager_execution()

'''
Execute training
'''

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='707 Project')
    parser.add_argument("action", type = str,
                        choices=["train", "continue", "test"])
    parser.add_argument("--num_episodes", type = int, default = 100)
    args = parser.parse_args()

    Config = Configuration()
    if args.action == "train":
        MyTrainer = Trainer(Config, restore = False)
        MyTrainer.train()
        MyTrainer.plot_test_result()
    elif args.action == "continue":
        MyTrainer = Trainer(Config, restore = True)
        MyTrainer.train()
        MyTrainer.plot_test_result()
    elif args.action == "test":
        MyTrainer = Trainer(Config, restore = True)
        MyTrainer.test(args.num_episodes)

