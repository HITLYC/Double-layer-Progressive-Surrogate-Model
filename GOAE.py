import torch
import torch.nn as nn
import numpy as np
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('Agg')

def evaluateMSE(model, X, target):
    predict = model(X)
    MSE = np.sum(np.power((predict-target),2))/len(target)
    return MSE

# Encoder
class Encoder(nn.Module):
  def __init__(self, n_in, n_hidden, n_latent):
    super().__init__()
    self.linear1 = nn.Linear(n_in, n_hidden)
    self.linear2 = nn.Linear(n_hidden, n_latent)
    self.relu = nn.LeakyReLU(0.2)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.linear2(out)
    z = self.relu(out)
    return z

# Decoder
class Decoder(nn.Module):
  def __init__(self, n_latent, n_hidden, n_out):
    super().__init__()
    self.linear1 = nn.Linear(n_latent, n_hidden)
    self.linear2 = nn.Linear(n_hidden, n_out)
    self.relu = nn.LeakyReLU(0.2)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.linear2(out)
    w = self.sigmoid(out)
    return w

# GOAEModel
class GOAEModel(nn.Module):
    def __init__(self, n_in, n_hidden,n_latent):
        super().__init__()
        self.encoder = Encoder(n_in, n_hidden, n_latent)
        self.decoder = Decoder(n_latent, n_hidden, n_in)
      
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
if __name__ == '__main__':
    
    # Load Data
    train0Data = np.loadtxt('..//result0Train.csv', delimiter = ',')[:,1:-1]
    valid0Data = np.loadtxt('..//result0Valid.csv', delimiter = ',')[:,1:-1]
    test0Data = np.loadtxt('..//result0Test.csv', delimiter = ',')[:,1:-1]
    
    train1Data = np.loadtxt('..//result1Train.csv', delimiter = ',')[:,1:-1]
    valid1Data = np.loadtxt('..//result1Valid.csv', delimiter = ',')[:,1:-1]
    test1Data = np.loadtxt('..//result1Test.csv', delimiter = ',')[:,1:-1]
    
    # Training Set
    trainX0 = torch.tensor(train0Data, dtype = torch.float32).cuda()
    trainX00 = (1-trainX0).round()
    trainX1 = torch.tensor(train1Data, dtype = torch.float32).cuda()
    
    # Valid Set
    validX0 = torch.tensor(valid0Data, dtype = torch.float32).cuda()
    validX1 = torch.tensor(valid1Data, dtype = torch.float32).cuda()
    
    # Test Set
    testX0 = torch.tensor(test0Data, dtype = torch.float32).cuda()
    testX00 = (1-testX0).round()
    testX1 = torch.tensor(test1Data, dtype = torch.float32).cuda()
                 
    MSESave = []
    
    # Hyperparameter settings
    epochs = 5000
    batch = 64
    steps = int(len(train1Data)/batch)
    lr = 0.0034
    step_size = 500
    gamma = 0.97
    alpha = 3
    
    for t in range(0,5):
        train0Data = np.loadtxt('..//result0Train.csv', delimiter = ',')[:,1:-1]
        
        # Define GOAE model
        GOAE = GOAEModel(8, 6, 1).cuda()
        step_size = 100
        gamma = 1
        
        # Optimizer
        optimizer = torch.optim.Adam(AE.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
     
        MSEValidBest = 2
        
        # Train
        for epoch in range(epochs):
            np.random.shuffle(train0Data)
            np.random.shuffle(train1Data)
            
            train0X = np.vstack((train0Data, train0Data[0:len(train1Data)-len(train0Data)]))
            train1X = copy.deepcopy(train1Data)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Lambda Cal
                MSEout = torch.mean(torch.abs(trainX00-AE(trainX0))).item()
                MSEin = torch.mean(torch.abs(trainX1-AE(trainX1))).item()
                
                dout = 1/(1+math.exp(-alpha*MSEout))
                din = 1/(1+math.exp(-alpha*MSEin))
                
                lamout = dout/(dout+din)
                lamin = din/(dout+din)
                
                X0 = torch.tensor(train0X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y0 = AE(X0)
                X1 = torch.tensor(train1X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y1 = AE(X1)
                
                # Loss
                loss = lamin*torch.mean((X1-Y1)**2) - lamout*torch.mean((X0-Y0)**2)
                loss.backward()
                optimizer.step()
    
                MSEValid = (torch.mean((validX1-AE(validX1))**2) - torch.mean((validX0-AE(validX0))**2)).item()
                if MSEValid < MSEValidBest:
                    MSEValidBest = copy.deepcopy(MSEValid)
                    GOAESave = copy.deepcopy(GOAE)
                    
            scheduler.step()
        
        torch.save(GOAESave,'GOAE1-'+str(t)+'.pkl')    
        MSETest0 = torch.mean((testX0-GOAESave(testX0))**2).item()
        MSETest1 = torch.mean((testX1-GOAESave(testX1))**2).item()

        print(1, t, MSETest0, MSETest1)
        MSESave.append([1, t, MSETest0, MSETest1])
        
    for t in range(0,5):
        train0Data = np.loadtxt('..//result0Train.csv', delimiter = ',')[:,1:-1]
        valid0Data = np.loadtxt('..//result0Valid.csv', delimiter = ',')[:,1:-1]
        
        # Define GOAE model
        GOAE = GOAEModel(8, 6, 2).cuda()
        step_size = 500
        gamma = 0.97
        
        # Optimizer
        optimizer = torch.optim.Adam(AE.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
        
        MSEValidBest = 2
        
        # Train
        for epoch in range(epochs):
            np.random.shuffle(train0Data)
            np.random.shuffle(train1Data)
            
            train0X = np.vstack((train0Data, train0Data[0:len(train1Data)-len(train0Data)]))
            train1X = copy.deepcopy(train1Data)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Lambda Cal
                MSEout = torch.mean(torch.abs(trainX00-AE(trainX0))).item()
                MSEin = torch.mean(torch.abs(trainX1-AE(trainX1))).item()
                
                dout = 1/(1+math.exp(-alpha*MSEout))
                din = 1/(1+math.exp(-alpha*MSEin))
                
                lamout = dout/(dout+din)
                lamin = din/(dout+din)
                
                X0 = torch.tensor(train0X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y0 = AE(X0)
                X1 = torch.tensor(train1X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y1 = AE(X1)
                
                loss = lamin*torch.mean((X1-Y1)**2) - lamout*torch.mean((X0-Y0)**2)
                loss.backward()
                optimizer.step()
    
                MSEValid = (torch.mean((validX1-AE(validX1))**2) - torch.mean((validX0-AE(validX0))**2)).item()
                           
                if MSEValid < MSEValidBest:
                    MSEValidBest = copy.deepcopy(MSEValid)
                    GOAESave = copy.deepcopy(GOAE)
                    
            scheduler.step()
        
        torch.save(GOAESave,'GOAE2-'+str(t)+'.pkl')    
        MSETest0 = torch.mean((testX0-GOAESave(testX0))**2).item()
        MSETest1 = torch.mean((testX1-GOAESave(testX1))**2).item()

        print(2, t, MSETest0, MSETest1)
        MSESave.append([2, t, MSETest0, MSETest1])
   
    for t in range(0,5):
        train0Data = np.loadtxt('..//result0Train.csv', delimiter = ',')[:,1:-1]
        valid0Data = np.loadtxt('..//result0Valid.csv', delimiter = ',')[:,1:-1]
        
        # Define GOAE model
        GOAE = GOAEModel(8, 13, 3).cuda()
        step_size = 2500
        gamma = 0.94
        
        # Optimizer
        optimizer = torch.optim.Adam(AE.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
        
        MSEValidBest = 2
        
        # Train
        for epoch in range(epochs):
            np.random.shuffle(train0Data)
            np.random.shuffle(train1Data)
            
            train0X = np.vstack((train0Data, train0Data[0:len(train1Data)-len(train0Data)]))
            train1X = copy.deepcopy(train1Data)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Lambda Cal
                MSEout = torch.mean(torch.abs(trainX00-AE(trainX0))).item()
                MSEin = torch.mean(torch.abs(trainX1-AE(trainX1))).item()
                
                dout = 1/(1+math.exp(-alpha*MSEout))
                din = 1/(1+math.exp(-alpha*MSEin))
                
                lamout = dout/(dout+din)
                lamin = din/(dout+din)
                
                X0 = torch.tensor(train0X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y0 = AE(X0)
                X1 = torch.tensor(train1X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y1 = AE(X1)
                
                loss = lamin*torch.mean((X1-Y1)**2) - lamout*torch.mean((X0-Y0)**2)
                loss.backward()
                optimizer.step()
    
                MSEValid = (torch.mean((validX1-AE(validX1))**2) - torch.mean((validX0-AE(validX0))**2)).item()
                           
                if MSEValid < MSEValidBest:
                    MSEValidBest = copy.deepcopy(MSEValid)
                    GOAESave = copy.deepcopy(GOAE)
                    
            scheduler.step()
        
        torch.save(GOAESave,'GOAE3-'+str(t)+'.pkl')    
        MSETest0 = torch.mean((testX0-GOAESave(testX0))**2).item()
        MSETest1 = torch.mean((testX1-GOAESave(testX1))**2).item()

        print(3, t, MSETest0, MSETest1)
        MSESave.append([3, t, MSETest0, MSETest1])
        
    for t in range(0,5):
        train0Data = np.loadtxt('..//result0Train.csv', delimiter = ',')[:,1:-1]
        valid0Data = np.loadtxt('..//result0Valid.csv', delimiter = ',')[:,1:-1]
        
        # Define GOAE model
        GOAE = GOAEModel(8, 4, 4).cuda()
        step_size = 500
        gamma = 0.995
        
        # Optimizer
        optimizer = torch.optim.Adam(AE.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)

        MSEValidBest = 2
        
        # Train
        for epoch in range(epochs):
            np.random.shuffle(train0Data)
            np.random.shuffle(train1Data)
            
            train0X = np.vstack((train0Data, train0Data[0:len(train1Data)-len(train0Data)]))
            train1X = copy.deepcopy(train1Data)
            
            for step in range(steps):
                optimizer.zero_grad()
                
                # Lambda Cal
                MSEout = torch.mean(torch.abs(trainX00-AE(trainX0))).item()
                MSEin = torch.mean(torch.abs(trainX1-AE(trainX1))).item()
                
                dout = 1/(1+math.exp(-alpha*MSEout))
                din = 1/(1+math.exp(-alpha*MSEin))
                
                lamout = dout/(dout+din)
                lamin = din/(dout+din)
                
                X0 = torch.tensor(train0X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y0 = AE(X0)
                X1 = torch.tensor(train1X[step*batch:(step+1)*batch,:], dtype = torch.float32).cuda()
                Y1 = AE(X1)
                
                loss = lamin*torch.mean((X1-Y1)**2) - lamout*torch.mean((X0-Y0)**2)
                loss.backward()
                optimizer.step()
    
                MSEValid = (torch.mean((validX1-AE(validX1))**2) - torch.mean((validX0-AE(validX0))**2)).item()
                           
                if MSEValid < MSEValidBest:
                    MSEValidBest = copy.deepcopy(MSEValid)
                    GOAESave = copy.deepcopy(GOAE)
                     
            scheduler.step()
        
        torch.save(GOAESave,'GOAE4-'+str(t)+'.pkl')    
        MSETest0 = torch.mean((testX0-GOAESave(testX0))**2).item()
        MSETest1 = torch.mean((testX1-GOAESave(testX1))**2).item()

        print(4, t, MSETest0, MSETest1)
        MSESave.append([4, t, MSETest0, MSETest1])
        t = t + 1
        
    np.savetxt('GOAE.csv',np.array(MSESave), delimiter = ',') # Save the experiment result