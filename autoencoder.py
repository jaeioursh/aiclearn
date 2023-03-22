import numpy as np
import pickle 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 


from aic.aic import aic

import pyximport
pyximport.install()
from teaming.cceamtl import one_agent

class Autoencoder(nn.Module):
    def __init__(self,q):
        super(Autoencoder, self).__init__()
        self.S1=400
        self.S2=100
        
        self.q=q
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Linear(self.q,self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S1,self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S2,2)
            #nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(2,self.S2),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S2,self.S1),
            nn.LeakyReLU(negative_slope=0.05),
            #nn.Sigmoid(),
            nn.Linear(self.S1,self.q)
            
        )
    
        self.apply(self._init_weights)
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def decode(self,x):
        x = self.decoder(x)
        return x

    def encode(self,x):
        x = self.encoder(x)
        return x


def train(model,data,fname="save/test.mdl", num_epochs=50000, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)#, 
                                 #weight_decay=1e-5) # <--
   

    for epoch in range(num_epochs):
        

        
        recon = model(data)
        loss = criterion(recon, data)
        print(epoch,loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch%(num_epochs//10)==0:
            print("saving")
            torch.save(model.state_dict(), fname)

def gen():
    with open("save/a.pkl","rb") as f:
        arry,p=pickle.load(f)



    params=[a[0][:4] for a in arry.flatten() if a is not None]
    shapes=[s.shape for s in params[0]]
    lens=[len(s.flatten()) for s in params[0]]
    idxs=np.cumsum(lens)
    data=np.array([np.concatenate([p.flatten() for p in s]).astype(np.float32) for s in params])
    dim=sum(lens)
    print(dim)
    print([s[:].shape for s in params[0]])

    data=torch.tensor(data)
    mdl=Autoencoder(dim)
    train(mdl,data)

def load():
    with open("save/a.pkl","rb") as f:
        arry,p=pickle.load(f)



    params=[a[0][:4] for a in arry.flatten() if a is not None]
    shapes=[s.shape for s in params[0]]
    lens=[len(s.flatten()) for s in params[0]]
    idxs=np.cumsum(lens)
    data=np.array([np.concatenate([p.flatten() for p in s]).astype(np.float32) for s in params])
    dim=sum(lens)
    print(dim)
    print([s[:].shape for s in params[0]])

    data=torch.tensor(data)
    mdl=Autoencoder(dim)
    PATH="save/test.mdl"
    mdl.load_state_dict(torch.load(PATH))

    latents=mdl.encode(data)
    latents=latents.detach().numpy().T

    env=aic(p)
    env.reset()
    print(p)
    agent=one_agent(env.state_size(),env.action_size(),20)

    fig1=plt.figure(1)
    plt.scatter(latents[0],latents[1])
    fig2=plt.figure(2)
    
    def test(event):
        
        fig2=plt.figure(2)
        fig2.clear()
        env.reset()
        A=[]
        x=event.xdata
        y=event.ydata

        x=torch.tensor([[x,y]],dtype=torch.float)
        ps=mdl.decode(x).detach().numpy()[0]
        weights=np.split(ps,idxs)
        ws=[]
        for w,s in zip(weights,shapes):
            ws.append(w.reshape(s))


        
        params=agent.__getstate__()
        for i in range(4):
            params[i][:]=ws[i]
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
        plt.title(str(g))
        fig2.canvas.draw()
        
    kind="button_press_event"
    #kind="motion_notify_event"
    fig1.canvas.mpl_connect(kind, test)


    plt.show()

    

if __name__ == "__main__":
    #gen()
    load()
