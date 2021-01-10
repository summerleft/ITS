# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import numpy as np

# hyper-params
H_NEURONS = 64
H_NEURONS2 = 16
LR_GAMMA = 0.99999
file_to_load = "/home/hjj/single/Train/save_net/dqn_test.h5"
# loss_out_file_path = "/home/hjj/single/Train/train_data/dqn_loss.txt"
# loss_out_file = open(loss_out_file_path, 'w')

class Net(nn.Module):
    def __init__(self, state_size, action_len, action_size):
        super(Net, self).__init__()
        out_size = action_len * action_size
        self.fc1 = nn.Linear(state_size, H_NEURONS)
        self.fc1.weight.data.normal_(0,0.1) # 权重初始化
        self.fc2 = nn.Linear(H_NEURONS,H_NEURONS2)
        self.out = nn.Linear(H_NEURONS2, out_size)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN:
    gamma = 0.9
    LR = 0.0001

    def __init__(self, state_size, action_mask, is_not_train=False,is_tensorboard=False):

        action_len = 1
        action_size = len(action_mask)
        self.out_size = action_len * action_size

        self.eval_net, self.target_net = Net(state_size, action_len, action_size), Net(state_size, action_len, action_size)
        
        self.training_step = 0
        self.writer = None
        self.mask = torch.FloatTensor(action_mask).unsqueeze(0)
        if is_not_train:
            self.loadModel(file_to_load)
        
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        #epsilon = 0.5 + 0.001 * self.epsilon_i
        #self.epsilon_i += 1
        if np.random.randn() <= 0.9:  # greedy policy
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy() # 如果action_value是一个矩阵，则输出每行的最大值的序号 #array([0])
            action = action[0]
        else: # random policy
            action = np.random.randint(0,self.out_size)
        return action


    def learn(self, s, a, r, s_):
        state = torch.FloatTensor(s)
        action = torch.tensor(a).view(-1,1)
        reward = torch.tensor(r).float().view(-1,1)
        next_state = torch.FloatTensor(s_)

        q_eval = self.eval_net(state)
        q_next = self.target_net(next_state).detach()
        q_target = reward + self.gamma * q_next.max(1)[0].view(-1,1)
        q_eval_target = torch.gather(q_eval, 1, action)
        loss = F.mse_loss(q_eval_target, q_target)
        
        print("#######################loss:    {:.8f}#####################\n".format(loss))
        # loss_out_file.write(format(loss))
        # loss_out_file.flush()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), 0.5) # 使梯度大小最大为0.5，以便连接参数不会一下子改变太多
        self.optimizer.step()

        self.training_step += 1
        #update the parameters
        if (self.training_step+1) % 20 == 0 and self.training_step > 200: #20 #每20回，将eval_net的参数更新到target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
            #if (self.training_step+1) % 20 == 0:
            self.LR *= LR_GAMMA
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.LR
        return

    # used for multi agents
    def returnSaveDict(self):
        dict_to_save = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict()
        }
        return dict_to_save

    def loadModel(self, file_to_load):
        dict_to_load = torch.load(file_to_load)
        self.eval_net.load_state_dict(dict_to_load['eval_net'])
        self.target_net.load_state_dict(dict_to_load['target_net'])
