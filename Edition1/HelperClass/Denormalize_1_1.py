import numpy as np
from HelperClass.Neuraltrain_1_1 import *
from HelperClass.DataReader_csv import *

class Denormalize_1_1(object):
    def __init__(self, net, reader):
        self.W_origin = net.W
        self.B_origin = net.B
        self.W_true = None
        self.B_true = None
        self.X_norm = reader.X_norm
        self.Y_norm = reader.Y_norm

    def Weights_denormalize(self):
        num_W = self.W_origin.shape[0]
        self.W_true = np.zeros((num_W,1))
        self.B_true = self.B_origin * self.Y_norm[0,1]
        for i in range(num_W):
            self.W_true[i,0] = self.W_origin[i,0] * self.Y_norm[0,1] / self.X_norm[i,1]
            self.B_true = self.B_true - self.W_true[i,0] * self.X_norm[i,0]
        #end for
        self.B_true = self.B_true  + self.Y_norm[0,0] 
        return self.W_true, self.B_true
