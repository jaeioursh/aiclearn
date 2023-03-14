import numpy as np
import tensorflow as tf
import numpy as np


from .logger import logger
from .ddpg import noise,agent

class learner:
    def __init__(self,team,sess):
        self.log=logger()
        self.nagents=len(team)
        self.dueling=0
        self.itr=0
        self.update_freq=5
        self.types=max(team)+1
        self.team=team
        s_dim, a_dim, hs_dim,ha_dim, lr, batch, gamma =[8,2,30,4,.01,32,.95]
        self.agents=[agent(sess,s_dim, a_dim, hs_dim,ha_dim, lr, batch,gamma,self.dueling) for i in range(self.types)]
        self.noise=[noise(a_dim) for i in range(self.types)]

    def act(self,S,idxs):
        A=[]
        for s,t in zip(S,self.team):
            i=idxs[t]
            a=self.agents[t].act(np.array([s]),i) + self.noise[t].sample()
            a=np.clip(a,-1,1)
            A.append(a[0])
        return A
    
    def store(self,S,A,R,SP,DONE,idxs):
        for s,a,sp,t in zip(S,A,SP,self.team):
            i=idxs[t]
            self.agents[t].store(s,a,R,sp,DONE,i)

    def learn(self):
        loss=[]
        for i in range(self.types):
            for j in range(2):
                L,_,_=self.agents[i].train_all(j)
                loss.append(L)
        return loss

    def randomize(self):
        self.team=np.random.randint(0,self.types,self.nagents)

    def save(self,fname="log.pkl"):
        print("saved")
        self.log.save(fname)

    def run(self,env,episode,render=False):
        s=env.reset()
        
        self.log.store("poi",np.array(env.data["Poi Positions"]),-1)
        if self.dueling:
            idxs=[episode%2]*self.types
            #idxs=np.random.randint(2,size=self.types) #which of the pair of policies to use
        else:
            idxs=[0]*self.types

        done=False
        R=[]
        #self.log.clear("position")
        while not done:
            self.log.store("position",np.array(env.data["Agent Positions"]),episode)
            self.itr+=1
            a=self.act(s,idxs)
            sp, r, done, info = env.step(a)
            if r[0]==0:
                r-=0.0
            r=r[0]
            R.append(r)
            self.store(s,a,r,sp,done,idxs)
            s=sp
            if self.itr%self.update_freq==0:
                L=self.learn()
                self.log.store("loss",L,episode)
            if render:
                env.render()
        
        self.log.store("reward",R)
        
        return R

    def test(self,env,itrs=100):
        
        old_team=self.team
        '''
        if self.dueling:
            idxs=[episode%2]*self.types
            #idxs=np.random.randint(2,size=self.types) #which of the pair of policies to use
        else:
        '''    
        idxs=[0]*self.types
        Rs=[]
        for i in range(itrs):
            self.randomize()
            s=env.reset()
            done=False
            R=[]
            while not done:
                
                a=self.act(s,idxs)
                sp, r, done, info = env.step(a)
                R.append(r[0])
                
                s=sp
            Rs.append(R)
        self.log.store("test",Rs)
        
        self.team=old_team

    def quick(self,env,episode,render=False):
        s=env.reset()
        
        for i in range(100):
            a=[[0,0] for i in range(self.nagents)]
            sp, r, done, info = env.step(a)
        return [0.0]
            
