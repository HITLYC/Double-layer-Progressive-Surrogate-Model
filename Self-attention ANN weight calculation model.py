
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random
import copy
import csv
import datetime

def evaluateMSE(model, X, target):
    predict = model(X)
    predict_n = np.squeeze(predict.cpu().detach().numpy(), 2)
    MSE = np.sum(np.power((predict_n-target),2))/len(target)
    return MSE

class SelfAttentionANN16(nn.Module):
    
    def __init__(self, n_feature, n_hidden, n_output, wq, wk, wv):
        super(SelfAttentionANN16, self).__init__()
        
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
        
        self.wq = nn.Parameter(wq)
        self.wk = nn.Parameter(wk)
        self.wv = nn.Parameter(wv)
        
    def self_attention(self, x, q, k, v):
        q = torch.unsqueeze(q.repeat(len(x), 1), 1)
        k = torch.unsqueeze(k.repeat(len(x), 1), 1)
        v = torch.unsqueeze(v.repeat(len(x), 1), 1)

        Q = torch.matmul(q, x)
        K = torch.matmul(k, x)
        V = torch.matmul(v, x)
        
        Kt = torch.transpose(K, 1, 2)
        Alpha = torch.matmul(Kt, Q)
        AlphaD = F.softmax(Alpha, dim = 2)
        
        O = torch.matmul(V, AlphaD)
        
        return O
        
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        
        o1 = self.self_attention(x[:, :, 0:16], self.wq[0], self.wk[0], self.wv[0])
        x1 = F.sigmoid(self.hidden(o1))
        
        x2 = self.output(x1)
        o2 = self.self_attention(x2, self.wq[1], self.wk[1], self.wv[1])
        o = torch.squeeze(o2, dim = 1)
        w = F.softmax(o, dim = 1)
        
        xt = x[:, :, 8:16]
        wt = torch.unsqueeze(w, 2)
        out = torch.matmul(xt, wt)
        
        return out

def Indicator(predict, target):
    tAve = np.average(target)
    MSE = np.average((predict-target)**2)
    MAE = np.average(abs(predict-target))
    RR = 1-np.sum((target-predict)**2)/np.sum((target-tAve)**2)
    STD = np.sum(((predict-tAve)**2/len(target))**0.5)
    MIN = np.min(abs(predict-target))
    MAX = np.max(abs(predict-target))
    return MSE, MAE, RR, STD, MIN, MAX
    
if __name__ == "__main__":
    
    # Load the predicting results of sub-models
    PredictResultTrain = np.loadtxt('PredictResultTrain.csv', delimiter = ',')
    X_train = torch.tensor(PredictResultTrain[:, 1:17], dtype=torch.float32).cuda()
    Y_train = PredictResultTrain[:, 17:18]
    
    PredictResultValid = np.loadtxt('PredictResultValid.csv', delimiter = ',')
    X_valid = torch.tensor(PredictResultValid[:, 1:17], dtype=torch.float32).cuda()
    Y_valid = PredictResultValid[:, 17:18]
    
    PredictResultTest = np.loadtxt('PredictResultTest.csv', delimiter = ',')
    X_test = torch.tensor(PredictResultTest[:, 1:17], dtype=torch.float32).cuda()
    Y_test = PredictResultTest[:, 17:18]
    
    metricsTest  = [['Method', 'No', 'MSE', 'MAE', 'RR', 'STD', 'MIN', 'MAX']]
    
    for t in range(1,5):
        TrainUse = copy.deepcopy(PredictResultTrain)
        
        # Hyperparameter settings for SelfAttentionANN
        lr = 30
        hidden = 29
        batch_size = 128
        step_size = 400
        gamma = 0.94
        
        steps = int(len(TrainUse)/batch_size)
        epochs = 3000
    
        MSEValidBest = 1
        
        wq = torch.randn(2)
        wk = torch.randn(2)
        wv = torch.randn(2)
        net = SelfAttentionANN16(16, hidden, 8, wq, wk, wv).cuda()
        params = net.parameters()
        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(params, lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
    
        for epoch in range(epochs):
            np.random.shuffle(TrainUse)
            X_train1 = TrainUse[:, 1:17]
            Y_train1 = TrainUse[:, 17:18]
    
            for step in range(steps):
                optimizer.zero_grad()
                
                data = torch.tensor(X_train1[step*batch_size:(step+1)*batch_size, :], dtype=torch.float32).cuda()
                label = torch.unsqueeze(torch.tensor(Y_train1[step*batch_size:(step+1)*batch_size, :], dtype=torch.float32).cuda(), 2)
    
                y = net(data)
                loss = loss_fn(y, label)
                loss.backward()
                optimizer.step()
            
                MSEValid = evaluateMSE(net, X_valid, Y_valid)

                if MSEValid < MSEValidBest :
                    MSEValidBest = MSEValid
                    SelfAttention16 = copy.deepcopy(net)
                    
            scheduler.step()
        
        SelfAttentionANN16PredictTest = np.squeeze(SelfAttention16(X_test).cpu().detach().numpy(), 2)
        MSE, MAE, RR, STD, MIN, MAX = Indicator(SelfAttentionANN16PredictTest, Y_test)
        metricsTest.append(['SelfAttentionANN16', t, MSE, MAE, RR, STD, MIN, MAX])
           
    f = open('SelfAttentionANN16(Test Set).csv','w',newline='')
    f_csv = csv.writer(f)
    f_csv.writerows(metricsTest)
    f.close()