from HelperClass2.DataReader_2_0 import DataReader_2_0
import csv
from pathlib import Path
import numpy as np

class DataReader_csv(DataReader_2_0):
    # read data from file
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = csv.reader(open(self.train_file_name))
            next(data)
            X=[]
            Y=[]
            for row in data:
                X.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
                if row[4] == 'Iris-setosa':
                    Y.append(1)
                elif row[4] == 'Iris-versicolor':
                    Y.append(2)
                else:
                    Y.append(3)
                #end if
            X=np.mat(X)
            Y=np.mat(Y)
            Y=Y.T
            self.XTrainRaw=X
            self.YTrainRaw=Y
            # assert(self.XTrainRaw.shape[0] == self.YTrainRaw.shape[0])
            self.num_train = self.XTrainRaw.shape[0]
            self.num_feature = self.XTrainRaw.shape[1]
            self.num_category = 3
            # this is for if no normalize requirment
            self.XTrain = self.XTrainRaw
            self.YTrain = self.YTrainRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if

        test_file = Path(self.test_file_name)
        if test_file.exists():
            data = csv.reader(open(self.test_file_name))
            next(data)
            X=[]
            Y=[]
            for row in data:
                X.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
                if row[4] == 'Iris-setosa':
                    Y.append(1)
                elif row[4] == 'Iris-versicolor':
                    Y.append(2)
                else:
                    Y.append(3)
                #end if
            X=np.mat(X)
            Y=np.mat(Y)
            Y=Y.T
            self.XTestRaw=X
            self.YTestRaw=Y
            # assert(self.XTestRaw.shape[0] == self.YTestRaw.shape[0])
            self.num_test = self.XTestRaw.shape[0]
            # this is for if no normalize requirment
            self.XTest = self.XTestRaw
            self.YTest = self.YTestRaw
            # in case there has no validation set created
            self.XDev = self.XTest
            self.YDev = self.YTest
        else:
            raise Exception("Cannot find test file!!!")
        #end if