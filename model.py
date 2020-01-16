import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # observation state to hidden state
        self.affine = nn.Linear(8, 128)
        # hidden state to action state
        self.action_layer = nn.Linear(128, 4)
        # value prediction for the state
        self.value_layer = nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        # observation state
        state = torch.from_numpy(state).float()
        # activation
        state = F.relu(self.affine(state))
        #observation
        state_value = self.value_layer(state)
        # action probablity distribution from actor critic
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        # sample an action from the prediction
        action = action_distribution.sample()
        
        # logrithmic probablity as per the actor critic algorithm
        self.logprobs.append(action_distribution.log_prob(action))
        # append observation states
        self.state_values.append(state_value)
        return action.item()
    
    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        # access latest rewards first
        for reward in self.rewards[::-1]:
            # update discounted rewards
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            # custom loss function for actor critic
            advantage = reward  - value.item()
            # logrithmic loss function
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            # cumulated loss for the episode
            loss += (action_loss + value_loss)   
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
