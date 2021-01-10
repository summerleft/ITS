# -*- coding: utf-8 -*-
import numpy as np
import traci
from ppo import PPO as Learner
from utils2 import get_loop_state, get_queue_state, get_head_waiting_time, get_avarage_waiting_time, get_laneID

#MEOMRY = 200
#BATCH = 64

class Agent:    
    def __init__(self, traci,is_not_train=False):
        self.controlTLIds = traci.trafficlight.getIDList()  # tuple ('0',)
        self.controlTLIds = self.controlTLIds[0] # string '0'
        self.phaseDefs = ['GrrrGrrr', 'rGrrrGrr', 'rrGrrrGr', 'rrrGrrrG']
        self.yelloPhases = ['yrrryrrr', 'ryrrryrr', 'rryrrryr', 'rrryrrry']
        action_mask = [1,1,1,1]
        self.detectorIDs = traci.inductionloop.getIDList()
        self.controlLanes = get_laneID(self.detectorIDs)
        self.reset()
        state_size = len(self.state)
        self.learner = Learner(state_size, action_mask,is_not_train)
        return
    
    # reset before each epoch
    def reset(self):
        self.prev_avg_wait_time = 0
        self.totalReward = 0
        self.totalCost = 0
        self.state = self._get_state()
        self.action = 1

    def _get_state(self):
        # 生成输入神经网络的state
        state = get_queue_state(self.controlLanes) + get_head_waiting_time(self.controlLanes)
        state = np.array(state)
        return state

    def setRYG(self,index,is_yello):
        light = self.controlTLIds
        if is_yello:
            ryg = self.yelloPhases[index]
        else:
            ryg = self.phaseDefs[index]
        traci.trafficlight.setRedYellowGreenState(light, ryg)

    def getPhase(self):
        light = self.controlTLIds
        phase_state = traci.trafficlight.getRedYellowGreenState(light)
        for i in range(len(self.phaseDefs)):
            if phase_state == self.phaseDefs[i]:
                phase = i
            if phase_state == self.yelloPhases[i]:
                phase = i
        return phase

    def agentAct(self):
        self.currentPhase = self.getPhase()
        self.action = self.learner.choose_action(self.state)
        return self.action,self.currentPhase

    def agentCulReward(self, is_sim=False):
        next_state = self._get_state()
        
        avg_wait_time = get_avarage_waiting_time(self.controlLanes)
        reward = self.prev_avg_wait_time - avg_wait_time
        self.prev_avg_wait_time = avg_wait_time

        self.totalReward += reward
        self.totalCost += avg_wait_time

        if not is_sim:
            self.learner.learn(self.state, self.action, reward, next_state)

        self.state = next_state
        return reward, avg_wait_time

    def printLog(self, outfile, simulation):
        print("************agent:simulation output*********")
        print("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.write("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.flush()