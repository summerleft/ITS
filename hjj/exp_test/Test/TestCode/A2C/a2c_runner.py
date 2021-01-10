# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import pandas as pd

import traci
from sumolib import checkBinary  # noqa
from utils import measure_traffic_step,collect_tripinfo

from agent import Agent


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--gui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

def main():
    traci.start(sumoCmd)

    agent = Agent(traci,True)
    traci.close()
    # start simulation
    traci.start(sumoCmd) 
    agent.reset()
    simulation_steps = 0
    traffic_data = []
    yell_time = 3
    atom_time = 30
    controlTLIds = "0"

    while simulation_steps < 60000:

        traffic_data.append(measure_traffic_step(controlTLIds,simulation_steps))
        agent.state = agent._getState()
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
             
        agent._getReward()  # stop training
        simulation_steps += atom_time
        # The END of simulationSteps loop

    traci.close()

    traffic_data = pd.DataFrame(traffic_data)
    traffic_data.to_csv("/home/hjj/exp_test/Test/TestData/near_unba/a2c/interval30/a2c_near_unba_30_new.csv")
    collect_tripinfo(trip_file,'/home/hjj/exp_test/Test/TestData/near_unba/a2c/interval30/a2c_near_unba_30_new')

if __name__ == "__main__":
    options = get_options()
    if options.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    trip_file = "tripinfo.xml"
    sumoConfig = "/home/hjj/exp_test/sumo_data/single_cross.sumocfg"
    sumoCmd = [sumoBinary, "-c", sumoConfig,"--tripinfo-output", trip_file]
    sumoCmd += ['--no-warnings', 'True']

    main()
    print("Completed!")