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

# _get_state()  #############################################
def get_loop_state(detectorIDs):
    # 输出：state（上个时隙通过该检测器的车辆平均速度）
    state = []
    car = []
    #detectorIDs = traci.inductionloop.getIDList()
    for detector in detectorIDs:
        speed = traci.inductionloop.getLastStepMeanSpeed(detector)
        state.append(speed)

        carID = traci.inductionloop.getLastStepVehicleIDs(detector)
        car.append(carID)
        car_num = traci.inductionloop.getLastStepVehicleNumber(detector)
    return state

def get_queue_state(controlLanes):
    # 输出：state 每条incoming车道的队伍长度
    state = []
    #controlLanes = get_laneID()
    for laneID in controlLanes:
        queue_length = traci.lane.getLastStepHaltingNumber(laneID)
        state.append(queue_length)
    return state

def get_head_waiting_time(controlLanes):
    # 输出：state 每条incoming车道最前面的车辆的等待时间
    state = []
    #controlLanes = get_laneID()
    for laneID in controlLanes:
        cars = traci.lane.getLastStepVehicleIDs(laneID)
        if len(cars) != 0:
            first_car = cars[-1]
            head_waiting_time = traci.vehicle.getAccumulatedWaitingTime(first_car)
        else:
            head_waiting_time = float(-1)
        state.append(head_waiting_time)
    return state
    # 有车，就是>=0的数；如果车道上没有车，就是-1

def get_laneID(detectorIDs):
    # 获取8条incoming车道的ID
    controlLanes = []
    for detector in detectorIDs:
        lanes = traci.inductionloop.getLaneID(detector)
        controlLanes.append(lanes)
    return controlLanes

###############################################################

def get_avarage_waiting_time(controlLanes):
    # 返回8条incoming车道上车辆的平均等待时间
    avg_waiting_time = 0
    total_waiting_time = 0
    total_car_num = 0
    #controlLanes = get_laneID()
    for laneID in controlLanes:
        cars = traci.lane.getLastStepVehicleIDs(laneID)
        total_car_num += len(cars)
        if len(cars) != 0:
            for car in cars:
                total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(car)
    if total_car_num != 0:
        avg_waiting_time = total_waiting_time/total_car_num
    return avg_waiting_time