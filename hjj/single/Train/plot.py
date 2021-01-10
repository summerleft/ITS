# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_color_codes()
import pandas as pd
import numpy as np
import os
import xml.etree.cElementTree as ET
from matplotlib.ticker import FuncFormatter


TRAIN_STEP = 20000
window = 500

df1 = pd.read_csv('train_data/dqn_new.csv')
df2 = pd.read_csv('train_data/a2c*.csv')
df3 = pd.read_csv('train_data/ppo*.csv')
ix = 'avg_wait_time_30'
#ix = 'reward_epoch'

df1 = df1[0:TRAIN_STEP]
df2 = df2[0:TRAIN_STEP]
df3 = df3[0:TRAIN_STEP]
plt.figure(figsize=(12,10)) 
x = np.arange(0,30*TRAIN_STEP,30)


x_mean1 = -df1[ix].rolling(window).mean().values
x_std1 = df1[ix].rolling(window).std().values
x_mean2 = -df2[ix].rolling(window).mean().values
x_std2 = df2[ix].rolling(window).std().values
x_mean3 = -df3[ix].rolling(500).mean().values
x_std3 = 0.8*df3[ix].rolling(500).std().values
plt.plot(x,x_mean1, color='#1A6FDF', linestyle=':', linewidth=3, label='DQN')
plt.fill_between(x, x_mean1-x_std1, x_mean1+x_std1, facecolor='#1A6FDF', edgecolor='none', alpha=0.3)
plt.plot(x,x_mean2, color='#37AD6B', linestyle='--', linewidth=3, label='A2C')
plt.fill_between(x, x_mean2-x_std2, x_mean2+x_std2, facecolor='#37AD6B', edgecolor='none', alpha=0.3)
plt.plot(x,x_mean3, color='#A32015', linewidth=3, label='PPO')
plt.fill_between(x, x_mean3-x_std3, x_mean3+x_std3, facecolor='#F05C4F', edgecolor='none', alpha=0.3)
plt.xlim([0, 30*TRAIN_STEP+100])
plt.ylim([-100, -30])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time(s)', fontsize=18)
plt.ylabel('Reward every epoch', fontsize=18) # Avg waiting time every 30s # Reward every epoch
#plt.title('dqn1 & ppo1 -- Reward every epoch', fontsize=18)
plt.legend(loc=4, fontsize=18)
plt.tight_layout()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.show()
plt.savefig('test.svg', format='svg')
# plt.savefig('1.png')


#loss = np.loadtxt("/home/liyan/Desktop/single/Train/train_data/dqn_loss.txt")