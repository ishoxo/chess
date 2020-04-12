#%%import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random
from torch import autograd
import numpy as np
from collections import deque
import torch
from torch.nn import MSELoss as MSE_loss


class ConvDQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ConvDQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)
        return self.buffer

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)




class DQNAgent:

    def __init__(self, experience=None, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.learning_rate = learning_rate
        self.experience = experience
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.model = ConvDQN(64, 4544)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state):
        state = autograd.Variable(torch.from_numpy(state).float().unsqueeze(0))
        qvals = self.model.forward(state)
        #print('qvals:', qvals)

        max_action = np.argmax(qvals.detach().numpy())
        r = random.uniform(0, 1)
        if r < self.epsilon:
            action = max_action
        else:
            random.shuffle(qvals)
            action = np.argmax(qvals.detach().numpy())
        return action

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.experience, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        #print('reward and shape', rewards, len(rewards))
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1) #INIITALLY WAS SQUEEZE(1)
       #print('curr_Q:', curr_Q, 'len curr_Q:', curr_Q.size())
        next_Q = self.model.forward(next_states)
        #print('next_Q:', next_Q, 'len next_Q:', next_Q.size())
        max_next_Q = torch.max(next_Q) ##initiall max(nextq, 1)
       #print('rewards.squeeze(1)', rewards.squeeze(1))
        first = rewards.squeeze(1)
        second = max_next_Q
        #print('first:', first, 'second:', second)

        expected_Q = rewards.squeeze(1) + (1 - dones) * self.gamma * max_next_Q
        #print(expected_Q)

        loss = self.MSE_loss(curr_Q, expected_Q.detach())
        return loss

    def update(self, batch_size):
        batch = self.sample(batch_size)
        #print('baatchy boy:', batch)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()