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

if __name__ == "__main__":
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
