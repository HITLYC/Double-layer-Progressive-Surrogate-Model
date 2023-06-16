
import torch
import torch.nn as nn
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import csv
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch import optim
import copy
import random
import datetime

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
    
# Load Data and Handle Them
def loadData(filePath, model, label):    
    data = np.loadtxt(filePath, delimiter = ',') 
    X = torch.tensor(data[:,1:-1], dtype = torch.float32) 
    comX = model.encoder(X).detach().numpy() 
      
    Y = label*np.ones([len(comX),1])
    
    return comX, Y

# MLP
class MLP(nn.Module):

    def __init__(self, inputDim, hiddenDim):
        super().__init__()
        self.fc = nn.Linear(inputDim, hiddenDim, bias=True)
        self.classify = nn.Linear(hiddenDim, 1, bias=True)
        self.act = torch.sigmoid

    def forward(self, x):
        out = self.act(self.fc(x))
        out = self.classify(out)
        result = torch.sigmoid(out)
        return result

# Metrics Computation
def compute_metrics(y_pred, y_true, model, dataset, dim, No):
    if y_pred.ndim == 2:
        y_pred = y_pred.flatten()
    if y_true.ndim == 2:
        y_true = y_true.flatten()
    part=y_pred^y_true 

    pcount=np.bincount(part) 
    tp_list=list(y_pred&y_true) 
    fp_list=list(y_pred& ~y_true)
    TP=tp_list.count(1) 
    FP=fp_list.count(1) 
    TN=pcount[0]-TP 
    FN=pcount[1]-FP 
    
    accuracy = (TP+TN)/(TP+TN+FP+FN) 
    if TP+FP == 0:
        precision = 0
    else:
        precision = TP / (TP+FP) 
    
    if TP+FN == 0:
        recall = 0
    else:
        recall = TP / (TP+FN) 
    F1 = (2*precision*recall) / (precision+recall)  
    
    return dim, No, model, dataset, TP, FP, FN, TN, accuracy, precision, recall, F1

if __name__ == '__main__':
    
    metricsTrain = [['dim','No.','model','dataset','TP','FP','FN','TN','accuracy','precision', 'recall', 'F1']]
    metricsValid = [['dim','No.','model','dataset','TP','FP','FN','TN','accuracy','precision', 'recall', 'F1']]
    metricsTest =  [['dim','No.','model','dataset','TP','FP','FN','TN','accuracy','precision', 'recall', 'F1']]
    for dim in range(1, 5):
        for No in range(1, 11):
            
            # Load GOAEModel
            GOAE = torch.load('GOAE'+str(dim)+'-'+str(No)+'.pkl')
            
            # Load Data
            XTrain0, YTrain0 = loadData('../result0Train.csv', GOAE, 0)
            XValid0, YValid0 = loadData('../result0Valid.csv', GOAE, 0)
            XTest0, YTest0 = loadData('../result0Test.csv', GOAE, 0)
            
            XTrain1, YTrain1 = loadData('../result1Train.csv', GOAE, 1)
            XValid1, YValid1 = loadData('../result1Valid.csv', GOAE, 1)
            XTest1, YTest1 = loadData('../result1Test.csv', GOAE, 1)
            
            # Training Data
            Train = np.vstack((np.hstack((XTrain0, YTrain0)), np.hstack((XTrain1, YTrain1)))) 
            np.random.shuffle(Train) 
            XTrain = Train[:,0:dim] 
            YTrain = Train[:,dim:dim+1]
            # Valid Data
            XValid = np.vstack((XValid0, XValid1))
            YValid = np.vstack((YValid0, YValid1))
            # Test Data
            XTest = np.vstack((XTest0, XTest1))
            YTest = np.vstack((YTest0, YTest1))
            
            # Hyperparameter settings of classifiers
            #%% dim = 1
            if dim == 1:
                #%% Naive Bayes
                clfBernoulli = BernoulliNB()
                clfGaussian = GaussianNB()
                #%% k-Nearest Neighbor
                clfKNN = KNeighborsClassifier(n_neighbors = 25)
                #%% Random Forest
                clfRFGINI = RandomForestClassifier(n_estimators = 3, max_depth = 1, criterion='gini', min_samples_split = 2, min_samples_leaf =1, 
                                                   max_features = 1)
                clfRFENTROPY = RandomForestClassifier(n_estimators = 3, max_depth = 1, criterion='entropy', min_samples_split = 2, min_samples_leaf =1, 
                                                   max_features = 1)
                #%% Decision Tree
                clfDTGINI = DecisionTreeClassifier(max_depth = 1, criterion='gini', min_samples_split = 2, min_samples_leaf =1, 
                                                   min_impurity_decrease = 0.0, max_features = 1)
                clfDTENTROPY = DecisionTreeClassifier(max_depth = 1, criterion='entropy', min_samples_split = 2, min_samples_leaf =1, 
                                                     min_impurity_decrease = 0.0, max_features = 1)
                #%% Support Vector Machine
                clfLinearSVC = LinearSVC(penalty = 'l1', loss = 'squared_hinge', C = 0.1, dual = False)
                clfSVC = SVC(C = 0.7, kernel = 'rbf')
                clfNuSVC = NuSVC(nu = 0.5, kernel = 'rbf')
                #%% Discriminant Analysis
                clfLinearDA = LinearDiscriminantAnalysis()
                clfQuadraticDA = QuadraticDiscriminantAnalysis(reg_param = 0.4)
                #%% AdaBoost
                clfSAMMEAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =1), n_estimators = 3, 
                                                learning_rate = 0.02, algorithm = 'SAMME')
                clfSAMMERAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =1), n_estimators = 2, 
                                                learning_rate = 0.007, algorithm = 'SAMME.R')
                #%% Logistic Regression
                clfLR = LogisticRegression()
                #%% MLP
                lr = 0.01
                hiddenDim = 12
                batch_size = 192
                step_size = 150
                gamma = 0.3
                epochs = 600
                
            #%% dim = 2
            if dim == 2:
                #%% Naive Bayes
                clfBernoulli = BernoulliNB()
                clfGaussian = GaussianNB()
                #%% k-Nearest Neighbor
                clfKNN = KNeighborsClassifier(n_neighbors = 46)
                #%% Random Forest
                clfRFGINI = RandomForestClassifier(n_estimators = 13, max_depth = 2, criterion='gini', min_samples_split = 2, min_samples_leaf = 1, 
                                                   max_features = 1)
                clfRFENTROPY = RandomForestClassifier(n_estimators = 20, max_depth = 1, criterion='entropy', min_samples_split = 2, min_samples_leaf = 1, 
                                                      max_features = 1)
                #%% Decision Tree
                clfDTGINI = DecisionTreeClassifier(max_depth = 1, criterion='gini', min_samples_split = 2, min_samples_leaf =1, 
                                                   min_impurity_decrease = 0.0, max_features = 1)
                clfDTENTROPY = DecisionTreeClassifier(max_depth = 1, criterion='entropy', min_samples_split = 2, min_samples_leaf =1, 
                                                     min_impurity_decrease = 0.0, max_features = 1)
                #%% Support Vector Machine
                clfLinearSVC = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.1, dual = True)
                clfSVC = SVC(C = 1.0, kernel = 'rbf')
                clfNuSVC = NuSVC(nu = 0.2, kernel = 'sigmoid')
                #%% Discriminant Analysis
                clfLinearDA = LinearDiscriminantAnalysis(solver = 'svd')
                clfQuadraticDA = QuadraticDiscriminantAnalysis(reg_param = 0.7)
                #%% AdaBoost
                clfSAMMEAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =1), n_estimators = 1, 
                                                learning_rate = 0.001, algorithm = 'SAMME')
                clfSAMMERAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =3), n_estimators = 10, 
                                                learning_rate = 0.2, algorithm = 'SAMME.R')
                #%% Logistic Regression
                clfLR = LogisticRegression()
                #%% MLP
                lr = 0.02
                hiddenDim = 8
                batch_size = 192
                step_size = 150
                gamma = 0.3
                epochs = 600
                
            #%% dim = 3
            if dim == 3:
                #%% Naive Bayes
                clfBernoulli = BernoulliNB()
                clfGaussian = GaussianNB()
                #%% k-Nearest Neighbor
                clfKNN = KNeighborsClassifier(n_neighbors = 24)
                #%% Random Forest
                clfRFGINI = RandomForestClassifier(n_estimators = 15, max_depth = 5, criterion='gini', min_samples_split = 20, min_samples_leaf = 9, 
                                                   max_features = 1)
                clfRFENTROPY = RandomForestClassifier(n_estimators = 29,  max_depth =5, criterion='entropy', min_samples_split = 2, min_samples_leaf = 8, 
                                                   max_features = 1)
                #%% Decision Tree
                clfDTGINI = DecisionTreeClassifier(max_depth = 4, criterion='gini', min_samples_split = 2, min_samples_leaf = 2, 
                                                   min_impurity_decrease = 0.0, max_features = 1)
                clfDTENTROPY = DecisionTreeClassifier(max_depth = 7, criterion='entropy', min_samples_split = 2, min_samples_leaf = 12, 
                                                     min_impurity_decrease = 0.0, max_features = 3)
                #%% Support Vector Machine
                clfLinearSVC = LinearSVC(penalty = 'l2', loss = 'hinge', C = 0.3, dual = True)
                clfSVC = SVC(C = 0.3, kernel = 'linear')
                clfNuSVC = NuSVC(nu = 0.1, kernel = 'rbf')
                #%% Discriminant Analysis
                clfLinearDA = LinearDiscriminantAnalysis(solver = 'svd')
                clfQuadraticDA = QuadraticDiscriminantAnalysis(reg_param = 0.0)
                #%% AdaBoost
                clfSAMMEAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =3), n_estimators = 300, 
                                                learning_rate = 0.2, algorithm = 'SAMME')
                clfSAMMERAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =1), n_estimators = 340, 
                                                learning_rate = 0.3, algorithm = 'SAMME.R')
                #%% Logistic Regression
                clfLR = LogisticRegression()
                #%% MLP
                lr = 0.4
                hiddenDim = 16
                batch_size = 128
                step_size = 250
                gamma = 0.95
                epochs = 600
                
             #%% dim = 4
            if dim == 4:
                #%% Naive Bayes
                clfBernoulli = BernoulliNB()
                clfGaussian = GaussianNB()
                #%% k-Nearest Neighbor
                clfKNN = KNeighborsClassifier(n_neighbors = 31)
                #%% Random Forest
                clfRFGINI = RandomForestClassifier(n_estimators = 17, max_depth = 9, criterion='gini', min_samples_split = 41, min_samples_leaf = 12, 
                                                   max_features = 2)
                clfRFENTROPY = RandomForestClassifier(n_estimators = 79, max_depth = 8, criterion='entropy', min_samples_split = 14, min_samples_leaf = 5, 
                                                   max_features = 2,)
                #%% Decision Tree
                clfDTGINI = DecisionTreeClassifier(max_depth = 1, criterion='gini', min_samples_split = 2, min_samples_leaf =1, 
                                                   min_impurity_decrease = 0.0, max_features = 2)
                clfDTENTROPY = DecisionTreeClassifier(max_depth = 1, criterion='entropy', min_samples_split = 2, min_samples_leaf =1, 
                                                     min_impurity_decrease = 0.0, max_features = 2)
                #%% Support Vector Machine
                clfLinearSVC = LinearSVC(penalty = 'l1', loss = 'squared_hinge', C = 0.8, dual = False)
                clfSVC = SVC(C = 0.8, kernel = 'rbf')
                clfNuSVC = NuSVC(nu = 0.2, kernel = 'rbf')
                #%% Discriminant Analysis
                clfLinearDA = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage = 0.5)
                clfQuadraticDA = QuadraticDiscriminantAnalysis(reg_param = 0.5)
                #%% AdaBoost
                clfSAMMEAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth = 4), n_estimators = 400, 
                                                learning_rate = 0.2, algorithm = 'SAMME', random_state = 2)
                clfSAMMERAB = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth =1), n_estimators = 370, 
                                                learning_rate = 0.1, algorithm = 'SAMME.R', random_state = 1)
                #%% Logistic Regression
                clfLR = LogisticRegression()
                #%% MLP
                lr = 0.02
                hiddenDim = 6
                batch_size = 384
                step_size = 150
                gamma = 0.3
                epochs = 600
                

            #%% Model Train
            clfBernoulli.fit(XTrain, YTrain.flatten())
            clfGaussian.fit(XTrain, YTrain.flatten())
            clfKNN.fit(XTrain, YTrain.flatten())
            clfRFGINI.fit(XTrain,YTrain.flatten())
            clfRFENTROPY.fit(XTrain,YTrain.flatten())
            clfDTGINI.fit(XTrain,YTrain.flatten())
            clfDTENTROPY.fit(XTrain,YTrain.flatten())
            clfLinearSVC.fit(XTrain,YTrain.flatten())
            clfSVC.fit(XTrain,YTrain.flatten())
            clfNuSVC.fit(XTrain,YTrain.flatten())
            clfLinearDA.fit(XTrain,YTrain.flatten())
            clfQuadraticDA.fit(XTrain,YTrain.flatten())
            clfSAMMEAB.fit(XTrain,YTrain.flatten())
            clfSAMMERAB.fit(XTrain,YTrain.flatten())
            clfLR.fit(XTrain,YTrain.flatten())
            
            steps = int(len(Train)/batch_size)
            accuValidBest = 0                    
            clfMLP = MLP(dim, hiddenDim).cuda()
            params = clfMLP.parameters()
            loss_Fn = BCELoss()
            optimizer = optim.SGD(params, lr = lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = step_size, gamma = gamma)
            Train = np.vstack((np.hstack((XTrain0, YTrain0)), np.hstack((XTrain1, YTrain1)))) 
            for epoch in range(epochs):
                np.random.shuffle(Train) 
                XTrain = Train[:,0:dim] 
                YTrain = Train[:,dim:dim+1] 
                
                for step in range(steps):
                    optimizer.zero_grad()
                    
                    XBatch =  torch.tensor(XTrain[step*batch_size:(step+1)*batch_size, :], dtype = torch.float32).cuda()
                    YBatch = torch.tensor(YTrain[step*batch_size:(step+1)*batch_size, :], dtype = torch.float32).cuda()
                    
                    YPred = clfMLP(XBatch)
                    loss = loss_Fn(YPred, YBatch)
                    loss.backward()
                    optimizer.step()
                    
                    PValid = np.round(clfMLP(torch.tensor(XValid, dtype = torch.float32).cuda()).cpu().detach().numpy())
                    accuValid = compute_metrics(PValid.astype('int'), YValid.astype('int'), 'MLP', 'valid', dim, No)[8]
                    if accuValid > accuValidBest:
                        accuValidBest = accuValid                        
                        clfMLPSave = copy.deepcopy(clfMLP)
                scheduler.step()

            # %% Metrics
            PTest = clfBernoulli.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'BernoulliNB', 'test', dim, No))
            PTest = clfGaussian.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'GaussianNB', 'test', dim, No))
            PTest = clfKNN.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'KNN', 'test', dim, No))
            PTest = clfRFGINI.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'RFGINI', 'test', dim, No))
            PTest = clfRFENTROPY.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'RFENTROPY', 'test', dim, No))
            PTest = clfDTGINI.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'DTGINI', 'test', dim, No))
            PTest = clfDTENTROPY.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'DTENTROPY', 'test', dim, No))
            PTest = clfLinearSVC.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'LinearSVC', 'test', dim, No))
            PTest = clfSVC.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'SVC', 'test', dim, No))
            PTest = clfNuSVC.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'SNuVC', 'test', dim, No))
            metricsTest.append([dim, No, 'NoneGP', 'test'])
            metricsTest.append([dim, No, 'RBFGP', 'test'])
            metricsTest.append([dim, No, 'MaternGP', 'test'])
            metricsTest.append([dim, No, 'RationalQuadraticGP', 'test'])
            PTest = clfLinearDA.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'LinearDA', 'test', dim, No))
            PTest = clfQuadraticDA.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'QuadraticDA', 'test', dim, No))
            PTest = clfSAMMEAB.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'SAMMEAB', 'test', dim, No))
            PTest = clfSAMMERAB.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'SAMMERAB', 'test', dim, No))
            PTest = clfLR.predict(XTest)
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'LR', 'test', dim, No))
            PTest = np.round(clfMLPSave(torch.tensor(XTest, dtype = torch.float32).cuda()).cpu().detach().numpy())
            metricsTest.append(compute_metrics(PTest.astype('int'), YTest.astype('int'), 'MLP', 'test', dim, No))
            
            print(datetime.datetime.today(), dim, No)
            
            # Save Metrics
            f = open('GOAE Metrics(Test Set).csv', 'w', newline = '')
            f_csv = csv.writer(f)
            f_csv.writerows(metricsTest)
            f.close()