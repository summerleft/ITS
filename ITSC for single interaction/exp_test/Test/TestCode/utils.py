import os
import sys
import numpy as np
import torch

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import traci

import xml.etree.cElementTree as ET
import subprocess
import pandas as pd

GREEN = 0
YELLOW = 1
RED = 2
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

def collect_tripinfo(trip_file,file_name):
    # read trip xml, has to be called externally to get complete file
    tree = ET.ElementTree(file=trip_file)
    trip_data = []
    for child in tree.getroot():
        cur_trip = child.attrib
        cur_dict = {}
        cur_dict['id'] = cur_trip['id']
        cur_dict['depart_sec'] = cur_trip['depart']
        cur_dict['arrival_sec'] = cur_trip['arrival']
        cur_dict['duration_sec'] = cur_trip['duration']
        cur_dict['wait_step'] = cur_trip['waitingCount']
        cur_dict['wait_sec'] = cur_trip['waitingTime']
        trip_data.append(cur_dict)
    trip_data = pd.DataFrame(trip_data)
    trip_data.to_csv('%s_trip.csv' % (file_name))
    # delete the current xml
    cmd = 'rm ' + trip_file
    subprocess.check_call(cmd, shell=True)

def get_laneID(detectorIDs):
    # 获取8条incoming车道的ID
    controlLanes = []
    for detector in detectorIDs:
        lanes = traci.inductionloop.getLaneID(detector)
        controlLanes.append(lanes)
    return controlLanes

def measure_traffic_step(controlTLIds,cur_sec):
    # runner.py调用
    # 输入：controlTLIds,simulation_step
    # 输出：cur_traffic
    #      当前时隙、在SUMO网络中的车辆数、此时隙进入网路的车辆数、此时隙到达目的地从SUMO网络中消失的车辆数、
    #      平均等待时间（自上次自上次速度小于0.1m/s以来，速度低于0.1m/s所花费的时间
    #      平均速度、等待队列的方差、等待队列的均值
    cars = traci.vehicle.getIDList()
    num_tot_car = len(cars)
    num_in_car = traci.simulation.getDepartedNumber()
    num_out_car = traci.simulation.getArrivedNumber()
    if num_tot_car > 0:
        avg_waiting_time = np.mean([traci.vehicle.getAccumula(car) for car in cars])
        avg_speed = np.mean([traci.vehicle.getSpeed(car) for car in cars])
    else:
        avg_speed = 0
        avg_waiting_time = 0
    # all trip-related measurements are not supported by traci,
    # need to read from outputfile afterwards
    queues = []
    detectorIDs = traci.inductionloop.getIDList()
    controlLanes = get_laneID(detectorIDs)
    for lane in controlLanes:
        queues.append(traci.lane.getLastStepHaltingNumber(lane)) # 返回给定车道上一个时间步的停止车辆总数。低于0.1m/s的速度被视为停止
    avg_queue = np.mean(np.array(queues))
    std_queue = np.std(np.array(queues))
    cur_traffic = {
                   'time_sec': cur_sec,
                   'number_total_car': num_tot_car,
                   'number_departed_car': num_in_car,
                   'number_arrived_car': num_out_car,
                   'avg_wait_sec': avg_waiting_time,
                   'avg_speed_mps': avg_speed,
                   'std_queue': std_queue,
                   'avg_queue': avg_queue}
    return cur_traffic