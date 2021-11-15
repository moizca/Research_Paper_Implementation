import gym
import numpy as np
from math import exp
from itertools import product
class BasicEnv(gym.Env):
  def __init__(self):
        self.no_aa=3
        self.demandd=[[4,4,4],[3,3,3],[2,2,2],[1,1,1],[6,6,6],[7,7,7]]
        self.demand=np.array(self.demandd)
        self.demand1=[[4,4,4],[3,3,3],[2,2,2],[1,1,1],[6,6,6],[7,7,7]]
        self.ddd=self.demand1[0][0]+self.demand1[1][0]+self.demand1[2][0]+self.demand1[3][0]+self.demand1[4][0]+self.demand1[5][0]+1
        self.ddd_list=list(range(self.ddd))
        self.perm = list(product(self.ddd_list, repeat = 3))
        self.ddd_list_actions=[0,1]
        self.perm_a=list(product(self.ddd_list_actions, repeat = 3))
        self.perm_actions=[]
        for self.ptp in self.perm_a:
          if sum(self.ptp)==1 or sum(self.ptp)==0:
            self.perm_actions.append(self.ptp)
        self.perm.sort()
        self.perm_actions.sort()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=(self.demand1[0][0]+self.demand1[1][0]+self.demand1[2][0]+ self.demand1[3][0]+ self.demand1[4][0]+ self.demand1[5][0])**3, shape=(1,), dtype=np.int)
        self.max_steps_per_episode = np.array(6)
        self.aaa=2.04
        self.bbb=0.24
        self.c=[200, 250, 300]
        self.c=np.array([self.c]).T
        self.cap_dist=1
        self.Len=12
  def step(self, act):
        self.act=act
        self.action=np.array(self.perm_actions[self.act]).reshape(3,1).astype(np.int)
        self.sstep=self.sstep+1
        self.state =np.array(self.state+np.array([self.demand[self.sstep-1]]).T-np.array(self.perm_actions[self.act]).reshape(3,1).astype(np.int))
        self.deprivation_cost=np.array(0)
        self.accessibility_based_delivery_cost=np.dot(self.action.reshape(3,), self.c.reshape(3,))
        for s in self.state:
          if s>=0:
            self.dep=exp(self.aaa)*(exp(self.bbb*self.Len)-1)*exp(self.bbb*self.Len)**int(s)
          else:
            self.dep=np.array(0)
          self.deprivation_cost=np.array(self.dep+self.deprivation_cost)
        if self.sstep==1:
          self.deprivation_cost=self.deprivation_cost+exp(self.aaa)*(exp(self.bbb*self.Len)-1)
        if self.sstep==self.max_steps_per_episode:
          self.terminal_penelty_cost=np.array(0)
          for s in self.state:
            if s>=0:
              self.term=exp(self.aaa)*(exp(self.bbb*self.Len)-1)*exp(self.bbb*self.Len)**int(s)
            else:
              self.term=np.array(0)
            self.terminal_penelty_cost=np.array(self.terminal_penelty_cost+self.term)
        if self.sstep==1:
          self.reward=np.array(-self.deprivation_cost-self.accessibility_based_delivery_cost)
        elif self.sstep==self.max_steps_per_episode:
          self.reward=np.array(-self.accessibility_based_delivery_cost-self.terminal_penelty_cost)
        else:
          self.reward=np.array(-self.accessibility_based_delivery_cost-self.deprivation_cost)
        if self.sstep==self.max_steps_per_episode:    
          self.done = True
        else:
          self.done = False
        self.info = {}
        self.stat=np.array(self.perm.index(tuple(self.state.reshape(3,)))).astype(np.int).reshape(1,)
        return self.stat, np.sum(self.reward.astype(np.float)), self.done, self.info
  def render(self):
        pass
  def reset(self):
        self.sstep=0
        self.state=np.array([[0], [0], [0]])
        self.stat=np.array(self.perm.index(tuple(self.state.reshape(3,)))).astype(np.int).reshape(1,)
        return self.stat