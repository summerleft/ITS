import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
 
#cur_dir = './output'

def calcu_total_wait(trip_file):
    print(trip_file)
    res = pd.read_csv(trip_file)
    cols = ['wait_sec','wait_step','duration_sec']
    for index in cols:
        print(index)
        print('mean: %.2f' % np.mean(res[index]))
        print('std: %.2f' % np.std(res[index]))
        

#calcu_total_wait('run_data/static_unbalanced_trip.csv')
#calcu_total_wait('run_data/dqn_unbalanced_trip.csv')
#calcu_total_wait('run_data/a2c_unbalanced_trip.csv')
#calcu_total_wait('run_data/ppo_unbalanced_trip.csv')

#calcu_total_wait('run_data/static_low_trip.csv')
#calcu_total_wait('run_data/dqn_low_trip.csv')

calcu_total_wait('/home/hjj/single_retrain/Train/dqn_retrain_trip.csv')
#calcu_total_wait('/home/hjj/single/Test/run_data/dqn_near_trip.csv')
#calcu_total_wait('/home/hjj/single/Test/run_data/ppo_near_trip.csv')

#calcu_total_wait('run_data/static_over_trip.csv')
#calcu_total_wait('run_data/dqn_over_trip.csv')


"""
./output/ppo_traffic_trip.csv # 每次都会有小的变化
wait_sec
mean: 97.74
std: 70.44
wait_step
mean: 1.62
std: 0.84
duration_sec
mean: 156.58
std: 74.03

./output/static_traffic_trip.csv # 不会改变
wait_sec
mean: 109.64
std: 130.22
wait_step
mean: 1.40
std: 1.15
duration_sec
mean: 164.97
std: 141.47
"""