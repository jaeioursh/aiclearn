import numpy as np
import pickle 
import matplotlib.pyplot as plt
import matplotlib as mpl

from aic.aic import aic
from tests import test1

import pyximport
pyximport.install()
from teaming.cceamtl import one_agent


with open("save/a.pkl","rb") as f:
    arry,p=pickle.load(f)


env=aic(p)
env.reset()
print(p)
agent=one_agent(env.state_size(),env.action_size(),20)




mp=[[0.0 if a is None else a[1] for a in arr] for arr in arry]
fig1=plt.figure(1)

plt.imshow(mp,cmap=mpl.colormaps["Reds"])
fig2=plt.figure(2)
XX,YY=-1,-1

def test(event):
    global XX,YY
    fig2=plt.figure(2)
    fig2.clear()
    env.reset()
    A=[]
    x=int(event.xdata+0.5)
    y=int(event.ydata+0.5)
    #if XX==x and YY==y:
    #    return
    XX=x
    YY=y
    print(y,x)
    params,r=arry[y,x]
    params=[np.copy(np.array(p)) for p in params]
    agent.__setstate__(params)

    for j in range(env.params.time_steps):
        S=env.state()[0]
        act=(np.array(agent.get_action(S))+1)/2
        A.append([env.agents[0].x,env.agents[0].y])
        env.action([act])
    g=sum(env.G())

    agents = np.array([[a[0], a[1]] for a in A])
    plt.plot(agents.T[0], agents.T[1], marker="o")

    pois = np.array([[p.x, p.y] for p in env.pois])
    colors = ["b", "k", "r", "c"]
    c = [colors[t] for t in env.poi_types]
    plt.scatter(pois.T[0], pois.T[1], marker="v", c=c)

    plt.xlim([-2, env.params.map_size + 2])
    plt.ylim([-2, env.params.map_size + 2])
    plt.title(str(g)+" "+str(r))
    fig2.canvas.draw()
    
kind="button_press_event"
#kind="motion_notify_event"
fig1.canvas.mpl_connect(kind, test)



plt.show()