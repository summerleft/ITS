# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

# hyper-params
H_NEURONS = 64
H_NEURONS2 = 16
LR_GAMMA = 0.99999
gamma = 0.9
file_to_load = "/home/hjj/exp_test/Test/TestCode/save_net/act_new.h5"

class Actor(nn.Module):
    def __init__(self, state_size, action_len, action_size):
        super(Actor, self).__init__()
        out_size = action_len * action_size
        self.action_len, self.action_size = action_len, action_size
        self.fc1 = nn.Linear(state_size, H_NEURONS)
        self.fc2 = nn.Linear(H_NEURONS, H_NEURONS2)
        self.action_head = nn.Linear(H_NEURONS2, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_output = self.action_head(x)
        action_probs = F.softmax(F.sigmoid(actor_output), dim = -1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        return action.squeeze().data.numpy(), action_dist


class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, H_NEURONS)
        self.fc2 = nn.Linear(H_NEURONS, H_NEURONS2)
        self.state_value = nn.Linear(H_NEURONS2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value

class A2C:
    def __init__(self, state_size, action_mask):
        action_len = 1
        action_size = len(action_mask)

        self.actor = Actor(state_size, action_len, action_size)
        self.critic = Critic(state_size)

        self.training_step = 0
        self.writer = None
        self.mask = torch.FloatTensor(action_mask).unsqueeze(0)
        
        self.loadModel(file_to_load)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = 0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 0.001)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        action,_ = self.actor(x)
        return action

    def log_prob_tensor(self, logits, actions):
        actions, log_pmf = torch.broadcast_tensors(actions, logits)
        actions = actions[..., :1]
        return log_pmf.gather(-1, actions).squeeze(-1)

    def learn(self, s,a,r,s_,len_of_track, no_of_workers):
        state = torch.unsqueeze(torch.FloatTensor(s), 0)
        action = torch.tensor(a)
        reward = torch.tensor(r).float()
        next_state = torch.unsqueeze(torch.FloatTensor(s_), 0)

        # discount_reward = torch.zeros_like(reward)
        
        # for j in range(no_of_workers):
        #     R = self.critic(state[0][j * len_of_track + len_of_track - 1])
        #     for i in reversed(range(0,len_of_track-1)):
        #         R = reward[i + j*len_of_track] + gamma * R
        #         discount_reward[i + j*len_of_track] = R
        #     discount_reward[j * len_of_track + len_of_track - 1] = reward[j * len_of_track + len_of_track - 1]

        _,action_dist = self.actor(state)
        logits = action_dist.logits
        
        log_prob = self.log_prob_tensor(logits, action)
        log_prob = log_prob.view(-1,1)

        entropy = action_dist.entropy().mean()

        v = self.critic(state)
        v_ = self.critic(next_state)
        target_v = reward + 0.9 * v_
        advantage = (target_v - v).detach()

        #print("log_prob",log_prob)
        #print("advantage",advantage)
        #print("log_prob * advantage",log_prob * advantage)

        actor_loss = -(log_prob * advantage).mean() - 0.01 * entropy
        #print("actor_loss %f"%actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        critic_loss = F.mse_loss(v, target_v) #F.mse_loss(v, discount_reward)
        #print("critic_loss %f"%critic_loss)
        self.critic_optimizer.zero_grad() # 重置梯度
        critic_loss.backward() # 计算反向传播
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5) # 使梯度大小最大为0.5，以便连接参数不会一下子改变太多
        self.critic_optimizer.step() # 更新连接参数
    
    # used for multi agents
    def returnSaveDict(self):
        dict_to_save = {
            'actor_net': self.actor.state_dict(),
            'critic_net': self.critic.state_dict()
        }
        return dict_to_save

    def loadModel(self, file_to_load):
        dict_to_load = torch.load(file_to_load)
        self.actor.load_state_dict(dict_to_load['actor_net'])
        self.critic.load_state_dict(dict_to_load['critic_net'])