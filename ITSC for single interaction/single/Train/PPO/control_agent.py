# -*- coding: utf-8 -*-
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

import torch
import xml.etree.cElementTree as ET
import subprocess

from agent import Agent

# save
# save_path = "/home/hjj/single/Train/save_net/ppo/act.h5"
# save_path_best = "/home/hjj/single/Train/save_net/ppo/best.h5"
out_file_path = "/home/hjj/single/Train/save_net/ppo/output.txt"
save_path_best = "/home/hjj/single/Train/save_net/ppo/best.h5"
save_path = "/home/hjj/exp_test/Test/TestCode/save_net/ppo_new.h5"

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

pre_total_cost = float('inf')
def save_multi_agents(save_path, save_path_best, agent):
    global pre_total_cost
    cul_total_cost = agent.totalCost
    dicts_to_save = agent.learner.returnSaveDict()
    torch.save(dicts_to_save, save_path)
           
    # save the best model
    if cul_total_cost < pre_total_cost:
        torch.save(dicts_to_save, save_path_best)
        pre_total_cost = cul_total_cost

def main():
    traci.start(sumoCmd)
    agent = Agent(traci,False)
    traci.close()

    TOTAL_EPOCH = 200
    SIM_STEPS = 6000
    atom_time = 30
    yell_time = 3
    outfile = open(out_file_path, 'w')
 
    start = time.clock()
    data = []
    for epoch in range(TOTAL_EPOCH):
        print("epoch:%i=================================================="%epoch)

        traci.start(sumoCmd) 
        agent.reset()
        simulation_steps = 0
        reward = 0
        total_reward = 0
        avg_wait_time = 0

        # simulation loop
        while simulation_steps < SIM_STEPS:
            action,current_phase = agent.agentAct()
            if action == current_phase:
                agent.setRYG(action,False)
                for j in range(atom_time):
                    traci.simulationStep()
            else:
                agent.setRYG(current_phase,True)
                for j in range(yell_time):
                    traci.simulationStep()
                agent.setRYG(action,False)
                for j in range(atom_time-yell_time):
                    traci.simulationStep()
                              
            reward, avg_wait_time = agent.agentCulReward(is_sim=False)
            total_reward += reward

            simulation_steps += atom_time

            agent.printLog(outfile, simulation_steps)
            print("Simulation {}: avg_wait_time_30 {}, reward_30 {}\n"
                        .format(simulation_steps, avg_wait_time, reward))
            log = {'Simulation_step': simulation_steps,'avg_wait_time_30':avg_wait_time,'reward_30': reward,'reward_epoch': total_reward}
            data.append(log)

        traci.close()
        save_multi_agents(save_path, save_path_best, agent)
    
    # The END of epoch loop
    df = pd.DataFrame(data)
    df.to_csv('/home/hjj/single/Train/train_data/ppo_new.csv')


if __name__ == '__main__':
    options = get_options()
    if options.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    trip_file = "tripinfo.xml"
    sumoConfig = "/home/hjj/exp_test/sumo_data/single_cross.sumocfg"
    sumoCmd = [sumoBinary, "-c", sumoConfig]#,"--tripinfo-output", trip_file]
    sumoCmd += ['--no-warnings', 'True']

    main()
    print("Completed!")