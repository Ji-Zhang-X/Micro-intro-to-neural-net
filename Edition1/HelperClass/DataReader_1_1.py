import numpy as np
import csv 
from pathlib import Path

class DataReader_1_1(object):
    def __init__(self, data_file):
        self.train_file_name = data_file
        self.num_train = 0
        self.num_feature = 0
        self.XTrain = None  
        self.YTrain = None  
        self.XRaw = None    
        self.YRaw = None    

    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = np.load(self.train_file_name, encoding='bytes', allow_pickle=True)
            self.num_train = self.XRaw.shape[0]
            self.num_feature = self.XRaw.shape[1]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

    def X_normalize(self):
        X_new = np.zeros(self.XRaw.shape)
        self.X_norm = np.zeros((self.num_feature,2))
        for i in range(self.num_feature):
            col_i = self.XRaw[:,i]
            max_value = np.max(col_i)
            min_value = np.min(col_i)
            self.X_norm[i,0] = min_value 
            self.X_norm[i,1] = max_value - min_value 
            new_col = (col_i - self.X_norm[i,0])/(self.X_norm[i,1])
            for j in range(self.num_train):
                X_new[j,i] = new_col[j]
        #end for
        self.XTrain = X_new

    def PredicationData_normalize(self, X):
        X_new = np.zeros(self.X_Raw.shape)
        n = X.shape[1]
        for i in range(n):
            col_i = X[:,i]
            X_new[:,i] = (col_i - self.X_norm[i,0]) / self.X_norm[i,1]
        return X_new

    def Y_normalize(self):
        self.Y_norm = np.zeros((1,2))
        max_value = np.max(self.YRaw)
        min_value = np.min(self.YRaw)
        self.Y_norm[0, 0] = min_value 
        self.Y_norm[0, 1] = max_value - min_value 
        y_new = (self.YRaw - min_value) / self.Y_norm[0, 1]
        self.YTrain = y_new

    def GetTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    def Disorganize(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        YP = np.random.permutation(self.YTrain)
        self.XTrain = XP
        self.YTrain = YP