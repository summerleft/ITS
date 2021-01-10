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


TRAIN_STEP = 40000
window = 1000

df1 = pd.read_csv('dqn_retrain.csv')
df2 = pd.read_csv('train_data/a2c*.csv')
df3 = pd.read_csv('train_data/ppo*.csv')
ix = 'avg_wait_time_30'
#ix = 'reward_epoch'

df1 = df1[0:TRAIN_STEP]
df2 = df2[0:TRAIN_STEP]
df3 = df3[0:TRAIN_STEP]
plt.figure(figsize=(12,10)) 
x = np.arange(0,30*TRAIN_STEP,30)


x_mean1 = df1[ix].rolling(window).mean().values
x_mean2 = df2[ix].rolling(window).mean().values
x_mean3 = df3[ix].rolling(window).mean().values
plt.plot(x,x_mean1, color='g', linestyle=':', linewidth=3, label='DQN')
plt.plot(x,x_mean2, color='b', linestyle='--', linewidth=3, label='A2C')
plt.plot(x,x_mean3, color='r', linewidth=3, label='PPO')
plt.xlim([0, 30*TRAIN_STEP+100])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time(s)', fontsize=18)
plt.ylabel('Avg waiting time every 30s', fontsize=18) # Avg waiting time every 30s # Reward every epoch
#plt.title('dqn1 & ppo1 -- Reward every epoch', fontsize=18)
plt.legend(loc='best', fontsize=18)
plt.tight_layout()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#plt.show()
plt.savefig("1.png")


#loss = np.loadtxt("/home/liyan/Desktop/single/Train/train_data/dqn_loss.txt")