import numpy as np
from HelperClass.Neuraltrain_1_1 import *
from HelperClass.DataReader_csv import *
from HelperClass.Denormalize_1_1 import *
from HelperClass.Visualization import *

file_name = "../Dataset/mlm.csv"

if __name__ == '__main__':
    reader = DataReader_csv(file_name)
    reader.ReadData()
    reader.X_normalize()
    reader.Y_normalize()
    hp = HyperParameters_1_0(2, 1, eta=0.01, max_epoch=5000, batch_size=10, eps = 1e-10)
    net = Neuraltrain_1_1(hp)
    net.train(reader, checkpoint=0.1)
    denormalize = Denormalize_1_1(net, reader)
    W_true, B_true = denormalize.Weights_denormalize()
    visual = Visualization(reader, W_true, B_true, 5000)
    visual.show_3D()
    print("W_true=")
    print(W_true)
    print("B_true=", B_true)
