#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import pandas as pd

from sumolib import checkBinary  # noqa
import traci  # noqa
from utils import measure_traffic_step,collect_tripinfo

# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_options():
    # action:存储方式，分为三种store、store_false、store_true
    # type:类型
    # dest:存储的变量
    # default:默认值
    # help:帮助信息
    optParser = optparse.OptionParser()
    optParser.add_option("--gui", action="store_true", default=False)
    options, args = optParser.parse_args()
    return options

def main():
    traci.start(sumoCmd)

    simulation_step = 0   
    atom_time = 30
    yell_time = 3
    traffic_data = []
    controlTLIds = "0"
    phaseDefs = ['GrrrGrrr', 'rGrrrGrr', 'rrGrrrGr', 'rrrGrrrG']
    yelloPhases = ['yrrryrrr', 'ryrrryrr', 'rryrrryr', 'rrryrrry']
    index = 0
    while simulation_step < 864000:

        ryg = yelloPhases[index % 4]
        traci.trafficlight.setRedYellowGreenState(controlTLIds, ryg)
        for j in range(yell_time):
            traci.simulationStep()

        ryg = phaseDefs[index % 4]
        traci.trafficlight.setRedYellowGreenState(controlTLIds, ryg)
        for j in range(atom_time-yell_time):
            traci.simulationStep()
        # for _ in range(ato_time):
        #     traci.simulationStep()
        simulation_step = simulation_step + atom_time
        index += 1

        traffic_data.append(measure_traffic_step(controlTLIds,simulation_step))

    traffic_data = pd.DataFrame(traffic_data)
    traffic_data.to_csv("/home/hjj/exp_test_v2/Test/TestData/real/static_real.csv")
    traci.close()
    collect_tripinfo(trip_file,'/home/hjj/exp_test_v2/Test/TestData/real/static_real')
    # 两个文件
    # run_data/static_unbalance.csv：从measure_traffic_step得到，包含每隔30秒记录的车辆数、平均等待时间、平均速度、队列方差及均值
    # run_data/static_unbalance_trip.csv：从"tripinfo.xml"中读出的数据，包含每辆车的到达时间、离开时间、旅行时间、等待次数、等待时间

if __name__ == "__main__":
    options = get_options()
    if options.gui:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    trip_file = "tripinfo_static.xml"
    sumoConfig = "/home/hjj/exp_test_v2/sumo_data/single_cross.sumocfg"
    sumoCmd = [sumoBinary, "-c", sumoConfig,"--tripinfo-output", trip_file]
    
    main()
    print("Completed!")
