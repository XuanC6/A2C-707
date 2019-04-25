# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from utils import conv2d, flattenallbut0, normc_initializer

#gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
tf.enable_eager_execution()

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

'''
Define the agent
'''
class Encoder(tf.keras.Model):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        #config for convnet feature extractors
        self.use_vision = self.config.use_vision
        self.height = self.config.height
        self.width = self.config.width
        self.kernel = self.config.kernel
        self.channles = self.config.channels
        self.filters = self.config.filters
        self.conv_dims = self.config.conv_dims
        self.convnet_kind = self.config.convnet_kind
        #config for lstm nets
        self.hist_len = self.config.history_length
        self.enc_units = self.config.enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       #return_sequences=True,
                                       #return_state=True,
                                       recurrent_initializer='glorot_uniform')
        #TODO: whether define a placeholder here or just convert inputs to tensor and feed
        #directly to the model which is a common practice for TF2.0
        #self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.hist_len]  + [
            #self.height, self.width])

    def extract_feature(self, x):
        """
        extract feature through conv nets
        :param x: shape should be (BS, Height, Width, Channel) aka "NWHC"
        :return: featur vectors in shape (BS, hidden_dim) hidden_dim is 256 for small
        kind convnet or  512  for large kind convnet
        """
        if self.convnet_kind == 'small':  # from A3C paper
            x = tf.nn.relu(conv2d(x, 16, [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 32, [4, 4], [2, 2], pad="VALID"))
            x = flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin',
                                           kernel_initializer=normc_initializer(1.0)))

        elif self.convnet_kind == 'large':  # Nature DQN
            x = tf.nn.relu(conv2d(x, 32, [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 64, [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 64, [3, 3], [1, 1], pad="VALID"))
            x = flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin',
                                           kernel_initializer=normc_initializer(1.0)))
        return x

    def call(self, inputs):
        if self.use_vision:
            inputs = self.extract_feature(inputs)
        output = self.gru(inputs)
        #output, state = self.gru(inputs, initial_state=hidden)

        #output shape (BS, timestes, units) units is the hidden size
        #hidden state shape (BS, units)
        #state is the last output output[:,-1,:] =  state
        return output

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))


class Option_Encoder(tf.keras.Model):

    def __init__(self, config):
        super(Option_Encoder, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        dense_dims = self.config.dense_dims_OE
        dense_activations = self.config.dense_activations_OE
        dense_initializer = self.config.dense_initializer_OE
        n_outputs = self.config.option_dim

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))
        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        input_tensor = tf.keras.layers.Input(shape = [self.config.input_length], batch_size = 1)
        self.call(input_tensor)


    def call(self, inputs):
        hidden = inputs
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        return self.output_layer(hidden)



class Critic(tf.keras.Model):

    def __init__(self, config):
        super(Critic, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        dense_dims = self.config.dense_dims_Cr
        dense_activations = self.config.dense_activations_Cr
        dense_initializer = self.config.dense_initializer_Cr
        n_outputs = 1

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        input_tensor = tf.keras.layers.Input(shape = [self.config.input_length], batch_size = 1)
        self.call(input_tensor)


    def call(self, inputs):
        hidden = inputs
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        return tf.squeeze(self.output_layer(hidden))



class Decoder(tf.keras.Model):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        self.gru_cell = tf.keras.layers.GRUCell(units = self.config.units)

        dense_dims = self.config.dense_dims_De
        dense_activations = self.config.dense_activations_De
        dense_initializer = self.config.dense_initializer_De
        n_outputs = self.config.output_dim_De

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        self.gru_cell.build(input_shape = (None, self.config.output_dim_De))

        input_tensor = tf.keras.layers.Input(shape = [self.config.units], batch_size = 1)
        hidden = input_tensor
        for layer in self.dense_layers:
            hidden = layer(hidden)
        _ =self.output_layer(hidden)


    def call(self, inputs, states):
        '''
        inputs: (1, n_actions)
        states: (1, option_dim)
        '''
        new_states, _ = self.gru_cell(inputs, [states])

        hidden = new_states
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        self.logits = self.output_layer(hidden)
        self.scores = tf.nn.softmax(self.logits)
        # self.action_tensor = tf.squeeze(tf.random.categorical(logits, 1))
        self.actions_entropy = -tf.reduce_sum(self.scores  * tf.math.log(self.scores))

        return new_states
