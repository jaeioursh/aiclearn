import numpy as np
import pyximport
pyximport.install()
from teaming.cceamtl import one_agent
from scipy import stats

from aic.aic import aic
from tests import test1

import pickle


def gen_dict(shape):
    arry=np.zeros(shape,dtype=object)
    arry[:]=None
    return arry

def pol2idx(info):
    idx=[]
    idx.append(stats.mode([i[0] for i in info]).mode[0])
    #for j in range(1,4):
    #    idx.append(int(np.mean([i[j] for i in info])*0.9999*10))
    idx.append(int(np.sum([ np.mean([i[j] for i in info])*0.9999*33 for j in range(1,4)])))
    return tuple(idx)

def pick(arry):
    vals=arry[arry!=None]
    return np.random.choice(vals,1)[0]
param=test1()
env=aic(param)
env.reset()
print(env.agents[0].x)
print(param.agent_pos)
agent=one_agent(env.state_size(),env.action_size(),20)

shape=(env.action_size()-3,100)
arry=gen_dict(shape)


for i in range(1000000):
    env.reset()
    A=[]
    if i>0:
        params,r=pick(arry)
        params=[np.copy(np.array(p)) for p in params]
        agent.__setstate__(params)
        agent.mutate()
    for j in range(env.params.time_steps):
        S=env.state()[0]
        act=(np.array(agent.get_action(S))+1)/2
        A.append([np.argmax(act[:-3]),act[-3],act[-2],act[-1]])
        env.action([act])
    g=sum(env.G())
    
    idx=pol2idx(A)

    info=arry[idx]
    if info is None or info[1]<g:
        params=[np.copy(np.array(p)) for p in agent.__getstate__()]
        arry[idx]=(params,g)

    if i%1000==0:
        print(i)
        with open("save/a.pkl","wb") as f:
            pickle.dump( [arry,param],f)

