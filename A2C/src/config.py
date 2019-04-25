# -*- coding: utf-8 -*-
import os
import gym
import tensorflow as tf
from datetime import datetime

#gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
tf.enable_eager_execution()

'''
All parameters and hyperparameters
'''

class Configuration:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/' + str(datetime.now()) + '/'

        self.env = gym.make("MsPacman-ram-v0")
        self.use_vision = False
        self.n_actions = self.env.action_space.n
        self.option_dim = 256
        self.input_length = 128

        '''
        Option_Encoder
        '''
        self.dense_dims_OE = [512, 512, 512]
        self.dense_activations_OE = ["elu"] * len(self.dense_dims_OE)
        self.dense_initializer_OE = "he_normal"

        '''
        Encoder
        '''
        self.history_length = 1
        self.batch_size = None
        #for ms-pacman
        self.enc_units = self.option_dim

        self.height = 210
        self.width  = 160
        self.kernel = None
        self.channels = 3
        self.filters = None
        self.conv_dims = None
        self.convnet_kind = 'small'

        '''
        Critic
        '''
        self.dense_dims_Cr = [512, 512, 512]
        self.dense_activations_Cr = ["elu"] * len(self.dense_dims_Cr)
        self.dense_initializer_Cr = "he_normal"

        '''
        Decoder
        '''
        self.units = self.option_dim
        self.dense_dims_De = [256, 256]
        self.dense_activations_De = ["elu"] * len(self.dense_dims_De)
        self.dense_initializer_De = "he_normal"

        self.output_dim_De = self.n_actions

        '''
        Trainer
        '''
        self.gamma = 0.99
        self.entropy_coeff = 1e-2
        self.N_compute_returns = 50
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        self.momentum = 0.95

        self.render_when_train = False
        self.render_when_test = False

        # self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum, use_nesterov=True)
        self.optimizer_actor = tf.train.AdamOptimizer(self.lr_actor)
        self.optimizer_critic = tf.train.AdamOptimizer(self.lr_critic)

        self.max_episodes = 10000
        self.save_interval = 50
        self.test_interval = 50
        self.n_test_episodes = 10

        '''
        log paths
        '''
        self.pic_dir = os.path.join(self.base_dir, "pic")
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)


        # self.weight_dir = os.path.join(self.base_dir, "weight")
        # if not os.path.exists(self.weight_dir):
        #     os.makedirs(self.weight_dir)
        # self.weight_path = self.weight_dir + '/saved_weights.ckpt'

        self.weight_enc_dir = os.path.join(self.base_dir, "weight_encoder")
        if not os.path.exists(self.weight_enc_dir):
            os.makedirs(self.weight_enc_dir)
        self.weight_enc_path = self.weight_enc_dir + '/saved_weights.ckpt'

        self.weight_oe_dir = os.path.join(self.base_dir, "weight_option_encoder")
        if not os.path.exists(self.weight_oe_dir):
            os.makedirs(self.weight_oe_dir)
        self.weight_oe_path = self.weight_oe_dir + '/saved_weights.ckpt'


        self.weight_cr_dir = os.path.join(self.base_dir, "weight_critic")
        if not os.path.exists(self.weight_cr_dir):
            os.makedirs(self.weight_cr_dir)
        self.weight_cr_path = self.weight_cr_dir + '/saved_weights.ckpt'


        self.weight_de_dir = os.path.join(self.base_dir, "weight_decoder")
        if not os.path.exists(self.weight_de_dir):
            os.makedirs(self.weight_de_dir)
        self.weight_de_path = self.weight_de_dir + '/saved_weights.ckpt'


        self.log_dir = os.path.join(self.base_dir, "logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.log_path = self.log_dir + '/train_logs.npz'
