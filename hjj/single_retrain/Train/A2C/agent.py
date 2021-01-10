# -*- coding: utf-8 -*-
import numpy as np
import traci
from a2c import A2C
from utils2 import get_loop_state, get_queue_state, get_head_waiting_time, get_avarage_waiting_time, get_laneID

#MEOMRY = 200
#BATCH = 64

class Agent:
    def __init__(self, name, traci):
        self.name = name
        action_mask = [1,1,1,1]
        self.controlTLIds = traci.trafficlight.getIDList()
        self.controlTLIds = self.controlTLIds[0]
        self.phaseDefs = ['GrrrGrrr', 'rGrrrGrr', 'rrGrrrGr', 'rrrGrrrG']
        self.yelloPhases = ['yrrryrrr', 'ryrrryrr', 'rryrrryr', 'rrryrrry']
        self.detectorIDs = traci.inductionloop.getIDList()
        self.controlLanes = get_laneID(self.detectorIDs,traci)
        self.reset(traci)
        state_size = len(self.state)
        if self.name == 'global':
            self.learner = A2C(state_size, action_mask)
        self.buffer_reset()
        return
    
    def reset(self,traci):
        self.prev_avg_wait_time = 0
        self.totalReward = 0
        self.totalCost = 0
        self.state = self._getState(traci)
        self.action = 1
    
    def buffer_reset(self):
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_s_ = []

    def setRYG(self,index,is_yello,traci):
        light = self.controlTLIds
        if is_yello:
            ryg = self.yelloPhases[index]
        else:
            ryg = self.phaseDefs[index]
        traci.trafficlight.setRedYellowGreenState(light, ryg)

    def getPhase(self,traci):
        light = self.controlTLIds
        phase_state = traci.trafficlight.getRedYellowGreenState(light)
        for i in range(len(self.phaseDefs)):
            if phase_state == self.phaseDefs[i]:
                phase = i
            if phase_state == self.yelloPhases[i]:
                phase = i
        return phase

    def agentAct(self,traci,global_agent):
        self.currentPhase = self.getPhase(traci)
        self.action = global_agent.learner.choose_action(self.state)
        self.action = int(self.action)
        
        return self.action,self.currentPhase
    
    def _getReward(self,traci):
        avg_wait_time = get_avarage_waiting_time(self.controlLanes,traci)
        reward = self.prev_avg_wait_time - avg_wait_time
        self.prev_avg_wait_time = avg_wait_time
        
        self.totalReward += reward
        self.totalCost += avg_wait_time
        return reward, avg_wait_time

    def _getState(self,traci):
        # 生成输入神经网络的state
        state = get_queue_state(self.controlLanes,traci) + get_head_waiting_time(self.controlLanes,traci)
        state = np.array(state)
        return state

    def printLog(self, outfile, simulation):
        print("************agent：simulation output*********")
        print("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.write("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.flush()