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

SIMULATION_STEP = 747
window = 20

df1 = pd.read_csv('low_over/static/interval30/static_low_over_30.csv')
df2 = pd.read_csv('low_over/dqn/interval30/dqn_low_over_30.csv')
df3 = pd.read_csv('low_over/a2c/interval30/a2c_low_over_30.csv')
df4 = pd.read_csv('low_over/ppo/interval30/ppo_low_over_30.csv')
ix = 'avg_queue'


df1 = df1[0:SIMULATION_STEP]
df2 = df2[0:SIMULATION_STEP]
df3 = df3[0:SIMULATION_STEP]
df4 = df4[0:SIMULATION_STEP]

x = np.arange(0, 30*SIMULATION_STEP, 30)


plt.figure(figsize=(12, 10))

x_mean1 = df1[ix].rolling(window).mean().values
x_mean2 = df2[ix].rolling(window).mean().values
x_mean3 = df3[ix].rolling(window).mean().values
x_mean4 = df4[ix].rolling(window).mean().values
plt.plot(x, x_mean1, color='r', linestyle='--', linewidth=3, label='static')
plt.plot(x, x_mean2, color='g', linestyle='-', linewidth=3, label='dqn')
plt.plot(x, x_mean3, color='b', linestyle=':', linewidth=3, label='a2c')
plt.plot(x, x_mean4, color='y', linestyle='-.', linewidth=3, label='ppo')
plt.xlim([0, 30*SIMULATION_STEP+30])
# plt.ylim([0, 25])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Time(s)', fontsize=18)
plt.ylabel('Avg queue length', fontsize=18)
plt.legend(loc='best', fontsize=18)
plt.tight_layout()
#plt.savefig('test.png')