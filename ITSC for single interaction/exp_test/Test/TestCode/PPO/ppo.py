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
file_to_load = "/home/hjj/exp_test/Test/TestCode/save_net/ppo*.h5"

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
        x = self.action_head(x)
        x = F.sigmoid(x)  # choose to add sigmoid to squeeze to probability or not
        return x.view(-1, self.action_len, self.action_size)


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


class PPO:
    gamma = 0.9
    clip_param = 0.2
    max_grad_norm = 0.5
    A_LR = 0.0001
    C_LR = 0.001

    def __init__(self, state_size, action_mask, is_not_train=False,is_tensorboard=False):

        action_len = 1
        action_size = len(action_mask)
        
        self.actor_net_old, self.actor_net_cur = Actor(state_size, action_len, action_size), Actor(state_size, action_len, action_size)
        self.critic_net = Critic(state_size)
        
        self.training_step = 0
        self.writer = None
        self.mask = torch.FloatTensor(action_mask).unsqueeze(0)

        if is_not_train:
            self.loadModel(file_to_load)  
        if is_tensorboard:
            self.writer = SummaryWriter('./exp')

        self.actor_optimizer = optim.Adam(self.actor_net_cur.parameters(), lr=self.A_LR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.C_LR)

        self.cur_log_dist = None
        self.old_log_dist = None

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = self.actor_net_cur(x)
        x = x * self.mask
        x = x.squeeze()
        m = Categorical(x)
        action = m.sample()
        self.cur_log_dist = m
        return action.squeeze().data.numpy()

    def learn(self, s, a, r, s_):
        state = torch.FloatTensor(s)
        action = torch.tensor(a)
        reward = torch.tensor(r).float()
        next_state = torch.FloatTensor(s_)

        v = self.critic_net(state)
        v_ = self.critic_net(next_state)
        target_v = reward + self.gamma * v_
        advantage = (target_v - v).detach()  # td_error

        old_log_dist = self.actor_net_old(state)
        old_log_dist = old_log_dist.squeeze()
        old_log_dist = Categorical(old_log_dist)
        old_log_prob = old_log_dist.log_prob(action)
        old_log_prob = old_log_prob.detach()

        ratio = torch.exp(self.cur_log_dist.log_prob(action) - old_log_prob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

        self.actor_net_old.load_state_dict(self.actor_net_cur.state_dict())

        # update actor
        entropy = self.cur_log_dist.entropy().mean()
        actor_loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy
        print("#######################actor_loss:    {:.8f}#####################\n".format(actor_loss))
        
        if self.writer:
            self.writer.add_scalar('loss/actor_loss', actor_loss, global_step=self.training_step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_net_cur.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic
        critic_loss = F.mse_loss(v, target_v)
        print("#######################critic_loss:    {:.8f}#####################\n".format(critic_loss))
        
        if self.writer:
            self.writer.add_scalar('loss/critic_loss', critic_loss, global_step=self.training_step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        self.old_log_dist = self.cur_log_dist
        self.training_step += 1

        if self.training_step % 20 == 0:
            self.A_LR *= LR_GAMMA
            self.C_LR *= LR_GAMMA
            for param_group in self.actor_optimizer.param_groups:
                param_group['lr'] = self.A_LR
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = self.C_LR
        return

    # used for multi agents
    def returnSaveDict(self):
        dict_to_save = {
            'actor_net_cur': self.actor_net_cur.state_dict(),
            'actor_net_old': self.actor_net_old.state_dict(),
            'critic_net': self.critic_net.state_dict()
        }
        return dict_to_save

    def loadModel(self, file_to_load):
        dict_to_load = torch.load(file_to_load)
        self.actor_net_cur.load_state_dict(dict_to_load['actor_net_cur'])
        self.actor_net_old.load_state_dict(dict_to_load['actor_net_old'])
        self.critic_net.load_state_dict(dict_to_load['critic_net'])