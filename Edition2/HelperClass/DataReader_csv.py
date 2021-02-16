from HelperClass.DataReader_1_1 import DataReader_1_1
import csv
from pathlib import Path
import numpy as np

class DataReader_csv(DataReader_1_1):
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            data = csv.reader(open(train_file))
            next(data)
            X=[]
            Y=[]
            for row in data:
                X.append([float(row[0]),float(row[1])])
                Y.append(float(row[2]))
            X=np.mat(X)
            Y=np.mat(Y)
            Y=Y.T
            self.XRaw=X
            self.YRaw=Y
            self.num_train = self.XRaw.shape[0]
            self.num_feature = self.XRaw.shape[1]
            self.XTrain = self.XRaw
            self.YTrain = self.YRaw
        else:
            raise Exception("Cannot find train file!!!")
        #end if