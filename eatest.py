from aic.parameter import parameter
from tests import test2
from aic.aic import aic
import pyximport
import tqdm
import numpy as np
from aic.view import view
pyximport.install()
from teaming.cceamtl import *
import matplotlib.pyplot as plt 

def display(data,env,n_steps,team):
    assignBestCceaPolicies(data,team)
    env.reset()
    pols=data["Agent Policies"]
    for j in range(n_steps):
        S=env.state()
        A=[]
        for s,pol in zip(S,pols):
            A.append((np.array(pol.get_action(s))+1)/2)
        env.action(A)
        g=sum(env.G())
        view(env,j,g)
def test(reward_type,disp=0):

    p=test2()
    env=aic(p)
    data=dict()
    data['Number of Agents']=p.n_agents
    data['Number of Types']=p.n_agents
    team=list(range(p.n_agents))
    data['Trains per Episode']=32
    data['Number of Policies']=data['Trains per Episode']
    initCcea(input_shape=env.state_size(), num_outputs=env.action_size(), num_units=20,num_types=p.n_agents)(data)

    gens=tqdm.trange(1000)
    info=[]
    for generation in gens:
        G=[]
        for i in range(data['Trains per Episode']):
            data["World Index"]=i
            assignCceaPolicies(data,team)
            env.reset()
            pols=data["Agent Policies"]
            for j in range(p.time_steps):
                S=env.state()
                A=[]
                for s,pol in zip(S,pols):
                    A.append((np.array(pol.get_action(s))+1)/2)
                env.action(A)
            g=sum(env.G())
            d=np.sum(env.D(),axis=1)
            if reward_type=="g":
                data["Agent Rewards"]=[g]*p.n_agents
            else:
                data["Agent Rewards"]=d

            rewardCceaPolicies(data,team)
            G.append(g)
        evolveCceaPolicies(data,team)
        if generation%250==-249:
            display(data,env,p.time_steps,team)
        info.append(max(G))
        gens.set_description("G:" +str(max(G)))
    if disp:

        if reward_type=="g":
            plt.title("G")
        else:
            plt.title("D")
        plt.plot(info)
        plt.show()
if __name__ == "__main__":
    test("g",1)