# -*- coding: utf-8 -*-
import numpy as np
import traci
from dqn import DQN as Learner
from utils2 import get_loop_state, get_queue_state, get_head_waiting_time, get_avarage_waiting_time, get_laneID

MEOMRY = 200
BATCH = 32

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
        self.buffer_reset()
        return
    
    # reset before each epoch
    def reset(self):
        self.prev_avg_wait_time = 0
        self.totalReward = 0
        self.totalCost = 0
        self.state = self._get_state()
        self.action = 1
        #self.buffer_reset()
    
    def buffer_reset(self):
        self.buffer_r = []
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_s_ = []
        self.buffer_count = 0

    def _get_state(self):
        # 生成输入神经网络的state
        state = get_queue_state(self.controlLanes) + get_head_waiting_time(self.controlLanes)
        state = np.array(state)
        return state

    def setRYG(self,index,is_yello): # control_agent:调用agent.setRYG(action,True/False)
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
        self.buffer_a.append(self.action)   
        return self.action,self.currentPhase

    def agentCulReward(self, is_sim=False):
        next_state = self._get_state()

        avg_wait_time = get_avarage_waiting_time(self.controlLanes)
        reward = self.prev_avg_wait_time - avg_wait_time
        self.prev_avg_wait_time = avg_wait_time
        
        self.totalReward += reward
        self.totalCost += avg_wait_time

        self.buffer_s.append(self.state)
        self.buffer_r.append(reward)
        self.buffer_s_.append(next_state)
        self.buffer_count += 1
        
        if not is_sim and self.buffer_count >= MEOMRY and self.buffer_count % 16 == 0:
            batch_mask = np.random.permutation(min(self.buffer_count,500))[:BATCH]
            bs, ba, br, bs_ = np.vstack(self.buffer_s)[batch_mask], np.vstack(self.buffer_a)[batch_mask], \
                                np.vstack(self.buffer_r)[batch_mask], np.vstack(self.buffer_s_)[batch_mask]
            self.learner.learn(bs, ba, br, bs_)
        if self.buffer_count >= 500: # 使得选取的buffer都是最近的1000个里面minibatch的
            del(self.buffer_a[0])
            del(self.buffer_s[0])
            del(self.buffer_s_[0])
            del(self.buffer_r[0])
        
        self.state = next_state
        return reward, avg_wait_time

    def printLog(self, outfile, simulation):
        print("************agent:printLog*********")
        print("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.write("Simulation {} group {}: totalCost {}, totalReward {}\n"
                      .format(simulation, self.controlTLIds, self.totalCost, self.totalReward))
        outfile.flush()