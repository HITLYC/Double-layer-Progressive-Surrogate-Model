
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch import optim
import random
import datetime
from sklearn.preprocessing import PolynomialFeatures
from joblib import load
import pickle
import copy
import math
import matplotlib.pyplot as plt
import time

# Encoder
class Encoder(nn.Module):
  def forward(self, w):
    out = self.linear1(w)
    out = self.linear2(out)
    z = self.relu(out)
    return z

# Decoder
class Decoder(nn.Module):
  def forward(self, z):
    out = self.linear1(z)
    out = self.linear2(out)
    w = self.sigmoid(out)
    return w

# GOAEModel
class GOAEModel(nn.Module):
    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y

# MLP(Classifier) 
class MLP(nn.Module):
    def forward(self, x):
        out = self.act(self.fc(x))
        out = self.classify(out)
        result = torch.sigmoid(out)
        return result
    
# RBF
class RBF(nn.Module):

    def kernal_fun(self, batches):
        n_input = batches.size(0)
        A = self.centers.view(self.num_centers, -1).repeat(n_input, 1, 1)
        B = batches.view(n_input, -1).unsqueeze(1).repeat(1, self.num_centers, 1)
        C = self.sigmas.view(self.num_centers, -1).repeat(n_input, 1, 1)
        D = -torch.div((A-B).pow(2), 2*C.pow(2))
        E = torch.exp(D.sum(2, keepdim = False))
        return E
    
    def forward(self, batches):
        radial_val = self.kernal_fun(batches)
        class_score = self.linear(radial_val)
        result = self.sigmoid(class_score)
        return result
    
# BP(Sub-model)
class BPNet(nn.Module):        
    def forward(self, x):
        x1 = torch.tanh(self.hidden1(x))
        x2 = torch.tanh(self.hidden2(x1))
        out = torch.sigmoid(self.output(x2))
        
        return out
    
# Save Kriging
class Employee:
    pass

# Kriging regpoly
def regpoly2(S):
    m, n = S.shape
    nn = int((n + 1)*(n + 2)/2) 
    
    f = np.hstack((np.ones((m, 1)), S, np.zeros((m, nn - n - 1))))
    j = n + 1
    q = n
    for k in range(n):
        f[:, np.arange(q) + j] = np.tile(S[:, k:k+1], (1, q)) * S[:, k:n]
        j = j + q
        q = q - 1

    return f

#  Kriging corr
def corrlin(theta, d):
    m, n = d.shape
    
    if type(theta) is np.ndarray:
        if len(theta) != n:
            raise ValueError('Length of theta must be 1 or '+ str(n))
    else:
        theta = np.tile(theta, (1, n))
        
    td = np.maximum(1-np.abs(d) * np.tile(theta, (m ,1)), 0)
    r = np.prod(td, axis = 1)
    
    return r

def correxp(theta, d):
    m, n = d.shape
    
    if type(theta) is np.ndarray:
        if len(theta) != n:
            raise ValueError('Length of theta must be 1 or '+ str(n))
    else:
        theta = np.tile(theta, (1, n))
        
    td = np.abs(d) * np.tile(-theta,(m,1))
    r = np.exp(np.sum(td, 1))
        
    return r

def corrspherical(theta, d):
    m, n = d.shape
    
    if type(theta) is np.ndarray:
        if len(theta) != n:
            raise ValueError('Length of theta must be 1 or '+ str(n))
    else:
        theta = np.tile(theta, (1, n))
        
    td = np.minimum(np.abs(d) * np.tile(theta, (m, 1)), 1)
    ss = 1 - td * (1.5 - 0.5 * td ** 2)
    r = np.prod(ss, axis = 1)
    
    return r

#  Kriging predictor
def predictor(dmodel, x):
    
    if (np.isnan(dmodel.beta.any())):
        y = float('nan')
        raise ValueError("DMODEL has not been found")
        
    m, n = dmodel.S.shape
    sx = x.shape
    if min(sx) == 1 & n > 1:
        nx = max(sx)
        if nx == n:
            mx = 1
            x = x.T
    else:
        mx = sx[0]
        nx = sx[1]
    if nx != n:
        raise ValueError("Dimension of trial sites should be " + str(n))
    
    # normalization
    x = (x - np.tile(dmodel.Ssc[0, : ], (mx, 1))) / np.tile(dmodel.Ssc[1, : ], (mx, 1))
    q =  dmodel.Ysc.shape[1] 
    y = np.zeros((mx, q)) 
 
    if mx == 1: 
        dx = np.tile(x, (m, 1)) - dmodel.S 
        
        f = dmodel.regr(x)
        r = dmodel.corr(dmodel.theta, dx)
        
        sy = np.dot(f, dmodel.beta) + np.dot(dmodel.gamma, r[:,np.newaxis]).T
        y = (dmodel.Ysc[0, :] + dmodel.Ysc[1, :] * sy).T
        
    else:
        dx = np.zeros((mx * m, n))
        kk = np.arange(m)
        for k in range(mx):
            dx[kk, :] = np.tile(x[k, :], (m,1)) - dmodel.S
            kk = kk + m
        
        f = dmodel.regr(x)
        r = dmodel.corr(dmodel.theta, dx)
        r = np.reshape(r, [m, mx], order = 'F')

        sy = np.dot(f, dmodel.beta) + np.dot(dmodel.gamma, r).T

        y = np.tile(dmodel.Ysc[0, :], (mx, 1)) + np.tile(dmodel.Ysc[1, :], (mx, 1)) * sy
    
    return y

# SelfAttentionANN16
class SelfAttentionANN16(nn.Module):        
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
        x1 = torch.sigmoid(self.hidden(o1))
        x2 = self.output(x1)
        o2 = self.self_attention(x2, self.wq[1], self.wk[1], self.wv[1])
        o = torch.squeeze(o2, dim = 1)
        w = F.softmax(o, dim = 1)
        
        xt = x[:, :, 8:16]
        wt = torch.unsqueeze(w, 2)
        out = torch.matmul(xt, wt)
        
        return out

# surrogateAnalysis model
def surrogateAnalysis(sample):
    # ------feasibility discrimination------
    # Load GOAE and classifier
    GOAEModel = torch.load('GOAE3.pkl').cuda()
    classifierModel = torch.load('3MLP.pkl').cuda()
    # compress sample
    X = torch.tensor(sample, dtype = torch.float32).cuda() 
    comX = GOAEModel.encoder(X) 
    # classification
    category = np.round(classifierModel(comX).cpu().detach().numpy())
    feasibility = np.hstack((sample, category))
    
    # ------analysis------
    # Load sub-models
    PRSModel = load('Polynomial.model') 
    PRS_reg = PolynomialFeatures(degree = 4) # PRS
    with open('kriging_regpoly2_correxp.pkl', 'rb') as f:
        KRG2ExpModel = pickle.load(f) # kriging_regpoly2_correxp
    with open('kriging_regpoly2_corrlin.pkl', 'rb') as f:
        KRG2LinModel = pickle.load(f) # kriging_regpoly2_corrlin
    with open('kriging_regpoly2_corrspherical.pkl', 'rb') as f:
        KRG2SphModel = pickle.load(f) # kriging_regpoly2_corrspherical
    RBFModel = torch.load('RBF.pkl') # RBF
    GBDTModel = load('GBDT.model') # GBDT
    XGBoostModel = load('XGBoost.model') # XGBoost
    BPModel = torch.load('BPNetwork.pkl') # BP
    # Load Self-attention ANN weight calculation model
    SelfAttentionANNModel = torch.load('SelfAttentionANN16.pkl')
    
    result = copy.deepcopy(feasibility) # result
    data = feasibility[feasibility[:, 8] == 1][:,0:8] # feasible samples
    dataTorch = torch.tensor(data, dtype=torch.float32)
    if data.shape[0] != 0:
        # ------sub-models predict------
        # 1
        data_poly = PRS_reg.fit_transform(data)
        PRSPredict = PRSModel.predict(data_poly)[:, np.newaxis]
        # 2
        KRG2ExpPredict = predictor(KRG2ExpModel, data)
        # 3
        KRG2LinPredict = predictor(KRG2LinModel, data)
        # 4
        KRG2SphPredict = predictor(KRG2SphModel, data)
        # 5
        RBFPredict = RBFModel(dataTorch).detach().numpy()
        # 6
        GBDTPredict = GBDTModel.predict(data)[:, np.newaxis]
        # 7
        XGBoostPredict = XGBoostModel.predict(data)[:, np.newaxis]
        # 8
        BPPredict = BPModel(dataTorch).detach().numpy()
        
        # total of sub-models
        predictTotal = np.hstack((data, PRSPredict, KRG2ExpPredict, KRG2LinPredict, KRG2SphPredict,  RBFPredict, GBDTPredict, 
                                   XGBoostPredict, BPPredict))
        predictTotal = torch.tensor(predictTotal, dtype=torch.float32).cuda()
        # prediction of hybrid
        predictResult = np.squeeze(SelfAttentionANNModel(predictTotal).cpu().detach().numpy(), 2)
        
        result[result[:, 8] == 1] = np.hstack((result[result[:, 8] == 1][:,0:8],predictResult))
        return result
    else:
        return result

if __name__ == "__main__":
   
    # Analysis of Spiral Shaft
    sC = np.array([[0.29281, 0.31579, 0.43333, 0.50000, 0.29800, 0.42857, 0.38333, 0.53659]])
    stress = surrogateAnalysis(sC)[:,8]
        