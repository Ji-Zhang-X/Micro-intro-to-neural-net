import numpy as np
import matplotlib.pyplot as plt

from HelperClass2.NeuralNet_2_2 import *
from HelperClass2.Visualizer_1_1 import *
from HelperClass2.DataReader_csv import *

train_data_name = "../Dataset/iris.csv"
test_data_name = "../Dataset/iris.csv"

if __name__ == '__main__':
    dataReader = DataReader_csv(train_data_name, test_data_name)
    dataReader.ReadData()
    dataReader.NormalizeY(NetType.MultipleClassifier, base=1)


    dataReader.NormalizeX()
    dataReader.Shuffle()
    dataReader.GenerateValidationSet()

    n_input = dataReader.num_feature
    n_hidden = 3
    n_output = dataReader.num_category
    eta, batch_size, max_epoch = 0.1, 10, 10000
    eps = 0.1

    hp = HyperParameters_2_0(n_input, n_hidden, n_output, eta, max_epoch, batch_size, eps, NetType.MultipleClassifier, InitialMethod.Xavier)
    net = NeuralNet_2_2(hp, "iris_43")

    # net.LoadResult()

    net.train(dataReader, 100, True)
    net.ShowTrainingHistory()

    print("W1=")
    print(net.wb1.W)
    print("B1=")
    print(net.wb1.B)
    print("W2=")
    print(net.wb2.W)
    print("B2=")
    print(net.wb2.B)
    