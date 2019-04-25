# -*- coding: utf-8 -*-
from collections import deque
import os
import sys
import random
import numpy as np
import tensorflow as tf
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Encoder, Option_Encoder, Critic, Decoder

#gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
tf.enable_eager_execution()

'''
Interact with the environment
Define loss function and train_op
'''

class Trainer:

    def __init__(self, config, restore):
        self.config = config

        #self.option_encoder = Option_Encoder(config)
        self.encoder = Encoder(config)
        self.critic = Critic(config)
        self.decoder = Decoder(config)

        self.env = config.env
        self.max_episodes = config.max_episodes
        self.render_when_train = config.render_when_train
        self.render_when_test = config.render_when_test
        self.gamma = config.gamma
        self.N_compute_returns = config.N_compute_returns

        self.hist_len = config.history_length

        self.optimizer_actor = config.optimizer_actor
        self.optimizer_critic = config.optimizer_critic

        self.save_interval = config.save_interval
        self.test_interval = config.test_interval

        self.n_episodes = np.asarray(0, dtype = int)
        # lists to store test results
        self.test_episodes = np.asarray([], dtype = int)
        self.test_reward_means = np.asarray([], dtype = float)
        self.test_reward_stds = np.asarray([], dtype = float)

        self.test_lifetime_means = np.asarray([], dtype = float)
        self.test_lifetime_stds = np.asarray([], dtype = float)

        self.n_test_episodes = config.n_test_episodes

        # self.weight_path = config.weight_path
        self.pic_dir = config.pic_dir

        self.weight_enc_path = config.weight_enc_path
        self.weight_oe_path = config.weight_oe_path
        self.weight_cr_path = config.weight_cr_path 
        self.weight_de_path = config.weight_de_path
        self.log_path = config.log_path

        self.restore = restore

        self.weights_initialized = False
        print(datetime.now())
        if os.path.isfile(self.weight_oe_path + '.index') and self.restore:
            # self.agent.load_weights(self.weight_path)
            #self.option_encoder.load_weights(self.weight_oe_path)
            self.encoder.load_weights(self.weight_enc_path)
            self.critic.load_weights(self.weight_cr_path)
            self.decoder.load_weights(self.weight_de_path)
            print('Weights Restored')
        else:
            print('Weights Initialized')
            self.weights_initialized = True


    def save_logs(self):
        np.savez(self.log_path, n_episodes = self.n_episodes, test_episodes = self.test_episodes, \
                test_reward_means = self.test_reward_means, test_reward_stds = self.test_reward_stds, \
                test_lifetime_means = self.test_lifetime_means, test_lifetime_stds = self.test_lifetime_stds)


    def train(self):
        if self.weights_initialized:
            self.test(self.n_test_episodes)
            self.save_logs()

            print("episode, actor_loss, critic_loss, mean_entropy, num_steps, replan_times, reward")
        
        if os.path.isfile(self.log_path) and self.restore:
            train_logs = np.load(self.log_path)

            self.n_episodes = train_logs["n_episodes"]
            self.test_episodes = train_logs["test_episodes"]
            self.test_reward_means = train_logs["test_reward_means"]
            self.test_reward_stds = train_logs["test_reward_stds"]
            self.test_lifetime_means = train_logs["test_lifetime_means"]
            self.test_lifetime_stds =train_logs["test_lifetime_stds"]

        while True:
            self.n_episodes += 1
            if self.n_episodes > self.max_episodes:
                break

            with tf.GradientTape(persistent= True) as tape:
                # 1. Generate an episode
                rewards, action_scores, state_value_tensors, state_values, entropys, n_steps, replan_times = \
                                                self.generate_episode(self.render_when_train)

                # convert to tensors
                action_scores = tf.identity(action_scores)
                state_value_tensors = tf.identity(state_value_tensors)
                entropys = tf.identity(entropys)

                # 2. Compute the returns G
                rewards /= 10.0
                # returns = self.compute_returns(rewards)
                returns = self.compute_returns_N(rewards, state_values)

                entropy_mean = tf.reduce_mean(entropys)
                loss_entropy = -self.config.entropy_coeff * entropy_mean

                loss_actor = -tf.reduce_mean((returns - state_values) * tf.log(action_scores)) + loss_entropy
                loss_critic =  tf.reduce_mean(tf.square(returns - state_value_tensors))

            actor_variables = self.encoder.variables + self.decoder.variables
            grads_actor = tape.gradient(loss_actor, actor_variables)
            self.optimizer_actor.apply_gradients(zip(grads_actor, actor_variables))

            grads_critic = tape.gradient(loss_critic, self.critic.variables)
            self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.variables))

            del tape
            
            print(self.n_episodes, loss_actor.numpy(), loss_critic.numpy(), entropy_mean.numpy(), n_steps, replan_times, np.sum(rewards), sep='\t')
            
            # Save data or do test
            if self.n_episodes % self.save_interval == 0:
                # self.agent.save_weights(self.weight_path)
                #self.option_encoder.save_weights(self.weight_oe_path)
                self.encoder.save_weights(self.weight_enc_path)
                self.critic.save_weights(self.weight_cr_path)
                self.decoder.save_weights(self.weight_de_path)
                
            if self.n_episodes % self.test_interval == 0:
                print()
                print(datetime.now())
                self.test(self.n_test_episodes)
                self.save_logs()

                print("episode, actor_loss, critic_loss, mean_entropy, num_steps, replan_times, reward")
            
        print(datetime.now())
        print('training finished')
        # self.agent.save_weights(self.weight_path)
        #self.option_encoder.save_weights(self.weight_oe_path)
        self.encoder.save_weights(self.weight_enc_path)
        self.critic.save_weights(self.weight_cr_path)
        self.decoder.save_weights(self.weight_de_path)


    def generate_episode(self, render, train=True):
        # Generates an episode.
        rewards = []
        action_scores = []
        state_value_tensors = []
        state_values = []
        entropys = []


        # step_this_option = 0
        n_steps = 0
        replan_times = 0
        action_onehot = tf.one_hot([0], self.config.output_dim_De)

        # replan = True

        # reset the environment and agent
        obs = self.env.reset()
        obs = self.preprocess_observation(obs)

        state_history = deque([np.zeros(obs.shape, dtype=np.float32)] * self.hist_len, self.hist_len)
        state_history.append(obs)

        if render:
            self.env.render()
        
        while True:
            replan_times += 1
            #states = self.option_encoder([obs])
            plan = self.encoder(tf.expand_dims(np.array(state_history, dtype=np.float32), 0))

            plan = self.decoder(action_onehot, plan)
            # step_this_option += 1
            
            if train:
                action_tensor = tf.squeeze(tf.random.categorical(self.decoder.logits, 1))
                action = action_tensor.numpy()
            else:
                action = np.argmax(np.squeeze(self.decoder.scores.numpy()))

            # avoid replanning forever
            # while True:
            #     action_tensor = tf.squeeze(tf.random.categorical(self.decoder.logits, 1))
            #     action = action_tensor.numpy()
            #     if step_this_option == 1 and action == self.config.n_actions:
            #         continue
            #     else:
            #         break

            # ##
            # if action == self.config.n_actions or step_this_option > self.config.max_n_decoding:
            #     replan = True
            #     step_this_option = 0
            #     continue
            # ##

            state_value_tensor = self.critic([obs])

            action_onehot = tf.one_hot([action], self.config.output_dim_De)
            action_score = tf.reduce_sum(self.decoder.scores*action_onehot)
            state_value = state_value_tensor.numpy()
            entropy_tensor = self.decoder.actions_entropy

            n_steps += 1
            next_obs, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()

            rewards.append(reward)
            action_scores.append(action_score)
            state_value_tensors.append(state_value_tensor)
            state_values.append(state_value)
            entropys.append(entropy_tensor)

            obs = self.preprocess_observation(next_obs)
            state_history.append(obs)
            
            if done:
                break

        return np.asarray(rewards), action_scores, state_value_tensors, \
                                np.asarray(state_values), entropys, n_steps, replan_times


    def compute_returns(self, rewards):
        # compute the return G
        T = len(rewards)
        returns = np.zeros((T))
        return_G = 0
        
        for t in reversed(range(T)):
            return_G = rewards[t] + self.gamma * return_G
            returns[t] = return_G
            
        return returns


    def compute_returns_N(self, rewards, state_values):
        # compute the return G(N_step)
        T = len(rewards)
        returns = np.zeros((T))
        
        for t in reversed(range(T)):
            
            if t + self.N_compute_returns >= T:
                Vend = 0
            else:
                Vend = state_values[t + self.N_compute_returns]

            signal = 0
            for k in range(self.N_compute_returns):
                if t+k < T:
                    reward = rewards[t+k]
                else:
                    reward = 0
                signal += (self.gamma**k) * reward

            returns[t] = signal + (self.gamma**self.N_compute_returns) * Vend
            
        return returns


    def test(self, n_test_episodes):
        # run certain test episodes on current policy, 
        # recording the mean/std of the cumulative reward.
        total_rewards = []
        lifetimes = []
        for _ in range(n_test_episodes):
            rewards, _, _, _, _, n_steps, _ = self.generate_episode(self.render_when_test, train = False)
            total_rewards.append(np.sum(rewards))
            lifetimes.append(n_steps)
            
        total_rewards = np.asarray(total_rewards)
        reward_mean = np.mean(total_rewards)
        reward_std = np.std(total_rewards)

        lifetimes = np.asarray(lifetimes)
        lifetime_mean = np.mean(lifetimes)
        lifetime_std =  np.std(lifetimes)
        
        print('episodes completed:', self.n_episodes)
        print('test reward mean over {} episodes:'.format(n_test_episodes), reward_mean)
        print('test reward std:', reward_std)

        print('test lifetime mean over {} episodes:'.format(n_test_episodes), lifetime_mean)
        print('test lifetime std:', lifetime_std)
        print('')
        
        self.test_episodes = np.append(self.test_episodes, self.n_episodes)

        self.test_reward_means = np.append(self.test_reward_means, reward_mean)
        self.test_reward_stds = np.append(self.test_reward_stds, reward_std)

        self.test_lifetime_means = np.append(self.test_lifetime_means, lifetime_mean)
        self.test_lifetime_stds = np.append(self.test_lifetime_stds, lifetime_std)


    def plot_test_result(self):
        plt.figure()
        plt.errorbar(self.test_episodes, self.test_reward_means, yerr = self.test_reward_stds)
        plt.title("total reward vs. the number of training episodes")
        plt.savefig(self.pic_dir + "/reward_error_bars.png")
        plt.clf()

        plt.figure()
        plt.plot(self.test_episodes, self.test_reward_means)
        plt.title("total reward vs. the number of training episodes")
        plt.savefig(self.pic_dir + "/reward_means.png")
        plt.clf()

        plt.figure()
        plt.errorbar(self.test_episodes, self.test_lifetime_means, yerr = self.test_lifetime_stds)
        plt.title("lifetime vs. the number of training episodes")
        plt.savefig(self.pic_dir + "/lifetime_error_bars.png")
        plt.clf()


    def preprocess_observation(self, obs):
        obs = obs.astype(np.float32)
        # normalize from -1. to 1.
        obs = (obs - 128) / 128.0
        return obs

