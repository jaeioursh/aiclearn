import re
import numpy as np
#import tensorflow as tf
import numpy as np
from copy import deepcopy as copy
from .logger import logger
import pyximport
from .cceamtl import *
from itertools import combinations
from math import comb
from collections import deque
from random import sample



def helper(t,k,n):
    if k==-1:
        return [t]
    lst=[]
    for i in range(n):
        if t[k+1]<=i:
            t[k]=i
            lst+=helper(copy(t),k-1,n)
    return lst

def s2z(s,i):
    s=s.copy()
    row=s[i,:].copy()
    s[i,:]=0
    z=np.vstack((row,s))
    return z.flatten()

def robust_sample(data,n):
    if len(data)<n: 
        smpl=data
    else:
        smpl=sample(data,n)
    return smpl

class learner:
    def __init__(self,nagents,types,skills,sim):
        self.graph=np.zeros((types,types,skills))
        self.log=logger()
        self.nagents=nagents
        self.itr=0
        self.types=types
        self.skills=skills
        self.team=[self.sample()]
        self.every_team=self.many_teams()
        self.test_teams=self.every_team
        sim.data["Number of Policies"]=32
        initCcea(input_shape=8, num_outputs=2, num_units=20,num_types=types*skills)(sim.data)
        

    def act(self,S,data,trial):
        policyCol=data["Agent Policies"]
        A=[]
        for s,pol in zip(S,policyCol):
  
            a = pol.get_action(s)*2.0
            A.append(a)
        return np.array(A)
    

    def randomize(self):
        length=len(self.every_team)
        teams=[]
        
        idx=np.random.choice(length)
        t=self.every_team[idx].copy()
        #np.random.shuffle(t)
        teams.append(t)
        self.team=teams
        #self.team=np.random.randint(0,self.types,self.nagents)

    def save(self,fname="log.pkl"):
        print("saved")
        self.log.save(fname)

    def assign(self,team,skills):
        return skills
        pols=[]
        for t,s in zip(team,skills):
            pols.append(self.skills*t+s)
        return pols

    def run(self,env,train_flag):
        populationSize=len(env.data['Agent Populations'][0])
        pop=env.data['Agent Populations']
        #team=self.team[0]
        skills=[0 for i in range(self.nagents)]
        G=[]
        #if train_flag==4:
        #    self.team=self.every_team
        for worldIndex in range(populationSize):
            env.data["World Index"]=worldIndex
            
            #for agent_idx in range(self.types):
            
            for team in self.team:
                s = env.reset() 
                done=False 
                #assignCceaPoliciesHOF(env.data)
                assignCceaPolicies(env.data,self.assign(team,skills))
                S,A=[],[]
                while not done:
                    self.itr+=1
                    
                    action=self.act(s,env.data,0)
                    S.append(s)
                    A.append(action)
                    s, r, done, info = env.step(action)
                
                pols=env.data["Agent Policies"] 
                
                        
                g=env.data["Global Reward"]
                d=env.data["Agent Rewards"]
                l=env.data["Local Rewards"]
              

                for i in range(len(s)):
                    pols[i].fitness=l[i,skills[i]]
                    #print(i,skills[i],l)
                    
                G.append(g)
            
            
        train_set = self.assign(team,skills)
        

        evolveCceaPolicies(env.data,train_set)

        self.log.store("reward",max(G))      
        return max(G)


 

    def put(self,key,data):
        self.log.store(key,data)


    def test(self,env,itrs=50,render=0):

        old_team=self.team
        #
        

        self.log.clear("position")
        self.log.clear("types")
        
        self.log.clear("poi")
        self.log.store("poi",np.array(env.data["Poi Positions"]))
        self.log.clear("poi vals")
        self.log.store("poi vals",np.array(env.data['Poi Static Values']))
        Rs=[]
        teams=copy(self.test_teams)
        print(teams)
        for i in range(len(teams)):

            
            
            #team=np.array(teams[i]).copy()
            #np.random.shuffle(team)
            self.team=[teams[i]]
            team=teams[i]
            #for i in range(itrs):
            assignBestCceaPolicies(env.data,team)
            #self.randomize()
            s=env.reset()
            done=False
            R=[]
            i=0
            self.log.store("types",self.team[0].copy(),i)
            
            while not done:
                
                self.log.store("position",np.array(env.data["Agent Positions"]),i)
                
                action=self.act(s,env.data,0)
                #action=self.idx2a(env,[1,1,3])
                #print(action)
                sp, r, done, info = env.step(action)
                if render:
                    env.render()
                
                s=sp
                i+=1
            g=env.data["Global Reward"]
            #d=env.data["Agent Reward"]
            #l=env.data["Local Reward"]
            Rs.append(g)
        self.log.store("test",Rs)
        
        self.team=old_team

    

    def quick(self,env,episode,render=False):
        s=env.reset()
        
        for i in range(100):
            a=[[0,0] for i in range(self.nagents)]
            sp, r, done, info = env.step(a)
        return [0.0]
            
    def many_teams(self):
        teams=[]
        C=comb(self.types,self.nagents)
        print("Combinations: "+str(C))
        if C<100:
            for t in combinations(range(self.types),self.nagents):
                teams.append(list(t))
        else:
            for i in range(50):
                teams.append(self.sample())

        return teams
    
    def sample(self):
        n,k=self.nagents,self.types
        return np.sort(np.random.choice(k,n,replace=False))


def test_net():
    a=Net()
    b=Net()
    x=np.array([[1,2,3,4,5,6,7,8]])
    y=np.array([[0]])
    print(a.feed(x))
    print(a.train(x,y))
    print(b.feed(x))
    print(b.train(x,y))

if __name__=="__main__":
    test_net()
    a=all_teams(5)
    print(a)
    
    