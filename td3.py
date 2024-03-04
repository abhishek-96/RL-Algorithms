'''
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Adopted from author's PyTorch Implementation
'''
# pylint: disable=C0103, R0913, R0901, W0221, R0902, R0914
import copy, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    '''
        The actor in TD3. Architecture from authors of TD3
    '''
    def __init__(self, state_dim, action_dim, max_action):
        nn.Module.__init__(self)
        self.max_action = max_action
        self.action_dim = action_dim
        
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        '''
            Returns the tanh normalized action
            Ensures that output <= self.max_action
        '''
        return self.max_action * self.layers(state)


class Critic(nn.Module):
    '''
        The critics in TD3. Architecture from authors of TD3
        We organize both critics within the same keras.Model
    '''
    def __init__(self, state_dim, action_dim):
        nn.Module.__init__(self)
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        for layer in self.q1:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)
        
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        for layer in self.q2:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        '''
            Returns the output for both critics. Using during critic training.
        '''
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)


    def q_1(self, state, action):
        '''
            Returns the output for only critic 1. Used to compute actor loss.
        '''
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3():
    '''
        The TD3 main class. Wraps around both the actor and critic, and provides
        three public methods:
        train_on_batch, which trains both the actor and critic on a batch of
        transitions
        select_action, which outputs the action by actor given a single state
        select_action_batch, which outputs the actions by actor given a batch
        of states.
    '''
    def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
    
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0


    def select_action(self, state):
        '''
            Select action for a single state.
            state: np.array, size (state_dim, )
            output: np.array, size (action_dim, )
        '''
        return self.actor(state).numpy().flatten()

    def select_action_batch(self, state):
        '''
            Select action for a batch of states.
            state: np.array, size (batch_size, state_dim)
            output: np.array, size (batch_size, action_dim)
        '''
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float)
        return self.actor(state).numpy()


    def train_on_batch(self, state, action, next_state, reward, not_done):
        '''
            Trains both the actor and the critics on a batch of transitions.
            state: tf tensor, size (batch_size, state_dim)
            action: tf tensor, size (batch_size, action_dim)
            next_state: tf tensor, size (batch_size, state_dim)
            reward: tf tensor, size (batch_size, 1)
            not_done: tf tensor, size (batch_size, 1)
            You need to implement part of this function.
        '''
        self.total_it += 1
        
        self.critic.train()
        self.actor.train()
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        self.critic_target.eval()
        self.actor_target.eval()
        
        # Select action according to policy and add clipped noise
        with torch.no_grad():
            noise = torch.clamp(torch.FloatTensor(np.random.normal(size=action.shape)) * self.policy_noise,
                                    -self.noise_clip, self.noise_clip)

            next_action = torch.clamp(self.actor_target(next_state) + noise,
                                        -self.max_action, self.max_action)
            print('Next Action Shape',next_action.shape)
            print('Next Action Val', next_action)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
        
            # Compute the target Q value
            y = reward + not_done * self.discount * target_Q
        
        # Get current Q estimates
        Q1, Q2 = self.critic(state, action)

        # Compute critic loss
        loss = F.mse_loss(Q1, y) + F.mse_loss(Q2, y)
        
        # Optimize the critic
        # loss_value.backward()
        loss.backward()
        self.critic_optimizer.step()
        
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor losses
            actions = self.actor(state)
            actor_loss = - torch.mean(self.critic.q_1(state, actions))
            
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target model
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)