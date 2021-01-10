# -*- coding: utf-8 -*-
# control_agent -> agent -> dqn
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import numpy as np
import pandas as pd

import traci
from sumolib import checkBinary

from agent import Agent

import torch
import xml.etree.cElementTree as ET
import subprocess

# save
save_path = "/home/hjj/single/Train/save_net/a2c/act.h5"
save_path_best = "/home/hjj/single/Train/save_net/a2c/best.h5"
out_file_path = "/home/hjj/single/Train/save_net/a2c/output.txt"


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--gui", action="store_true", default=False)
    options, args = optParser.parse_args()
    return options

def save_multi_agents(save_path, save_path_best, agent):
    dicts_to_save = agent.learner.returnSaveDict()
    torch.save(dicts_to_save, save_path)

def main():
    # no_of_workers = 1
    len_of_track = 16 # 一次MC更新的轨迹长度
    # 初始化workers
    workers = []
    traci.start(sumoCmd)

    global_agent = Agent('global', traci)
    traci.close(sumoCmd)

    TOTAL_EPOCH = 200
    SIM_STEPS = 6000
    atom_time = 30
    yell_time = 3
    outfile = open(out_file_path, 'w')
 
    start = time.clock()
    data = []
    Cmd = []
    
    for epoch in range(TOTAL_EPOCH):
        update_record = 0
        print("epoch%i========================================\n"%epoch)
        traci.start(sumoCmd)
        global_agent.reset(traci)
        simulation_steps = 0

        reward = 0
        total_reward = 0
        avg_wait_time = 0
        
        # 初始化结束
        
        # simulation loop
        while simulation_steps < SIM_STEPS:
            state = global_agent._getState(traci)
            global_agent.state = state # 重要！！！
            global_agent.buffer_s.append(state)
            action, current_phase = global_agent.agentAct(traci, global_agent) # worker[i]
            global_agent.buffer_a.append(action)
            if action == current_phase:
                global_agent.setRYG(action,False,traci)
                for j in range(atom_time):
                    traci.simulationStep()
            else:
                global_agent.setRYG(current_phase,True,traci)
                for j in range(yell_time):
                    traci.simulationStep()
                global_agent.setRYG(action,False,traci)
                for j in range(atom_time-yell_time):
                    traci.simulationStep()
            next_state = global_agent._getState(traci)
            global_agent.buffer_s_.append(next_state)
            reward, avg_wait_time = global_agent._getReward(traci)
            global_agent.buffer_r.append(reward)
            
            total_reward += reward

            print("Simulation {} : avg_wait_time_30 {}, reward_epoch {}\n"
                    .format(simulation_steps, avg_wait_time, total_reward))

            log = {'Simulation_step': simulation_steps, 'avg_wait_time_30':avg_wait_time,'reward_30': reward,'reward_epoch': total_reward}
            data.append(log)

            update_record += 1
            simulation_steps += atom_time
            if update_record % len_of_track == 0 or update_record % (SIM_STEPS/atom_time) == 0: # 一次轨迹为16或者到每个epoch的结尾时
                len_single_worker = len(global_agent.buffer_a)
                bs, ba, br, bs_ = np.vstack(global_agent.buffer_s), np.vstack(global_agent.buffer_a), np.vstack(global_agent.buffer_r), np.vstack(global_agent.buffer_s_)
                global_agent.learner.learn(bs, ba, br, bs_, len_single_worker, 1)

        traci.close()
        save_multi_agents(save_path, save_path_best, global_agent)

    # The END of epoch loop
    df = pd.DataFrame(data)
    df.to_csv('/home/hjj/single/Train/train_data/a2c_test.csv')


if __name__ == '__main__':
    starttime = time.time()
    options = get_options()
    if options.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    trip_file = "tripinfo.xml"
    sumoConfig = "/home/hjj/single/sumo_data/single_cross.sumocfg"
    sumoCmd = [sumoBinary, "-c", sumoConfig] #,"--tripinfo-output", trip_file]
    sumoCmd += ['--no-warnings', 'True']
    #sumoCmd += ['--waiting-time-memory', '500']
    main()
    print("Completed!")
    endtime = time.time()
    running_time = endtime-starttime
    print("Running Time: %.8s s" % running_time)