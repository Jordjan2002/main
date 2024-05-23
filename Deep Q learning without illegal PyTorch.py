import gym
import chess
import gym_chess
import random
from helpers import custom_reset, custom_step, find_max_Q_move, pos_to_index
import numpy as np
import copy
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import tensorflow as tf
import keras.backend as backend
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


device = torch.device("cuda:0")

input_shape = (8,8,3)
action_space_size_white = 36
action_space_size_black = 8

input_shape = (8,8,3)
action_space_size_white = 36
action_space_size_black = 8

# hyperparameters
episodes = 2_000
Replay_memory_size = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
target_network_update = 50 
Mini_batch_size = 64
discount = 0.99
start_epsilon_decaying = 1
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

MODEL_NAME_White = "DQNchess white"
MODEL_NAME_Black = "DQNchess black"
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20
Show_every = 50


class Net(nn.Module):

    def __init__(self, input_shape, action_space_size):
        super().__init__()
        self.input_shape = input_shape
        self.action_space_size = action_space_size
        
        self.conv1 = nn.Conv2d(1, 192, kernel_size=3)
        self.conv2 = nn.Conv2d(192, 100, kernel_size=3)
        self.conv3 = nn.Conv2d(100, 50, kernel_size=3)

        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512) #flattening.
        self.fc2 = nn.Linear(512, 36)

    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x


    def forward(self, x):
        x = self.convs(x)
        x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


net = Net().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


EPOCHS = 100
BATCH_SIZE = 10

def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)): # from 0, to the len of x, stepping BATCH_SIZE at a time. [:50] ..for now just to dev
            #print(f"{i}:{i+BATCH_SIZE}")
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            net.zero_grad()

            outputs = net(batch_X) # forward pass?
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()    # Does the update

        print(f"Epoch: {epoch}. Loss: {loss}")



class chessagentDQN():

    def __init__(self, model_name, input_shape, action_space_size):
        self.input_shape = input_shape
        self.action_space_size = action_space_size

        self.model = self.create() # policy network

        self.target_model = self.create() # target network
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=Replay_memory_size)
        self.target_update_counter = 0


    def create(self):
        model = Net(self.input_shape, self.action_space_size).to(device)

        return model
    
    def update_replay_memory(self, experience):
        self.replay_memory.append(experience)

    def get_Qs(self, state):
        return self.model.forward(np.array(state).reshape(-1, *state.shape))[0]    
    
    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, Mini_batch_size)

        current_states = np.array([experience[0] for experience in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.asarray([experience[3] for experience in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_Q = np.max(future_qs_list[index])
                new_q = reward + discount*max_future_Q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)


        X = torch.Tensor(X).to(device)
        Y = torch.Tensor(Y).to(device)

        net.zero_grad()

        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > target_network_update:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


