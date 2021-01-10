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
    no_of_workers = 1
    len_of_track = 16 # 一次MC更新的轨迹长度
    # 初始化workers
    workers = []
    traci.start(sumoCmd)
    for i in range(no_of_workers):
        i_name = 'W_%i' % i
        workers.append(Agent(i_name, traci))
    # 初始化global_agent为Agent类
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
        r = 0
        print("epoch%i========================================\n"%epoch)
        # 每个epoch都初始化
        for i in range(no_of_workers):
            Cmd = sumoCmd #+ ['--tripinfo-output', '%s_tripinfo.xml' % (workers[i].name)]
            # print(Cmd)
            traci.start(Cmd, label = "sim%i" %i)
        
        simulation_steps = 0
        
        # 初始化结束
        
        # simulation loop
        while simulation_steps < SIM_STEPS:
            c = 0
            for i in range(no_of_workers):
                traci.switch("sim%i"%i)
                state = workers[i]._getState(traci)
                workers[i].state = state # 重要！！！
                workers[i].buffer_s.append(state)
                action, current_phase = workers[i].agentAct(traci, global_agent) # worker[i]
                workers[i].buffer_a.append(action)
                if action == current_phase:
                    workers[i].setRYG(action,False,traci)
                    for j in range(atom_time):
                        traci.simulationStep()
                else:
                    workers[i].setRYG(current_phase,True,traci)
                    for j in range(yell_time):
                        traci.simulationStep()
                    workers[i].setRYG(action,False,traci)
                    for j in range(atom_time-yell_time):
                        traci.simulationStep()
                next_state = workers[i]._getState(traci)
                workers[i].buffer_s_.append(next_state)
                reward, avg_wait_time = workers[i]._getReward(traci)
                workers[i].buffer_r.append(reward)
                
                r += reward
                c += avg_wait_time

            print("Simulation {} : avg_wait_time_30 {}, reward_epoch {}\n"
                    .format(simulation_steps, c/no_of_workers, r/no_of_workers))

            log = {'Simulation_step': simulation_steps, 'avg_wait_time_30':c/no_of_workers,'reward_epoch': r/no_of_workers}
            data.append(log)
             

            update_record += 1
            simulation_steps += atom_time
            if update_record % len_of_track == 0 or update_record % (SIM_STEPS/atom_time) == 0: # 一次轨迹为16或者到每个epoch的结尾时
                #print(update_record)
                ba_list, br_list, bs_list, bns_list = [],[],[],[]
                for i in range(no_of_workers):
                    len_single_worker = len(workers[i].buffer_a)
                    #print(len_single_worker)
                    bs, ba, br, bs_ = np.vstack(workers[i].buffer_s), np.vstack(workers[i].buffer_a), np.vstack(workers[i].buffer_r), np.vstack(workers[i].buffer_s_)
                    ba_list += ba.tolist()
                    br_list += br.tolist()
                    bs_list += bs.tolist()
                    bns_list += bs_.tolist()

                    workers[i].buffer_a = []
                    workers[i].buffer_s = []
                    workers[i].buffer_r = []
                    workers[i].buffer_s_ = []
                
                bs = np.array(bs_list)
                ba = np.array(ba_list)
                br = np.array(br_list)
                bs_ = np.array(bns_list)
                global_agent.learner.learn(bs, ba, br, bs_, len_single_worker, no_of_workers)

                # print("global_agent actor parameters----------------------------------------------------")
                # for parameters in global_agent.learner.actor.parameters():
                #     print(parameters)

        for i in range(no_of_workers):
            traci.switch("sim%i"%i)
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
    running_time = starttime-endtime
    print("Running Time: %.8s s" % running_time)