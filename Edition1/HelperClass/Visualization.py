import matplotlib.pyplot as plt
import numpy as np


class Visualization(object):
    def __init__(self, datareader, W, B, epoch):
        self.X = datareader.XRaw
        self.Y = datareader.YRaw
        self.X_norm = datareader.X_norm
        self.Y_norm = datareader.Y_norm
        self.W = W
        self.B = B
        self.epoch = epoch
        self.fig = plt.figure()
        self.ax = self.fig.gca(fc='whitesmoke', projection='3d')

    def show_3D(self):
        plt.cla()
        self.ax.scatter(self.X[:, 0], self.X[:, 1], self.Y)
        x1 = np.linspace(self.X_norm[0,0], self.X_norm[0,0]+self.X_norm[0,1], num=150)
        x2 = np.linspace(self.X_norm[1,0], self.X_norm[1,0]+self.X_norm[1,1], num=150)
        X, Y = np.meshgrid(x1, x2)
        w1 = self.W[0,0]
        w2 = self.W[1,0]
        b = self.B[0,0]
        self.ax.plot_surface(X, Y, Z=X*w1+Y*w2+b, color="g", alpha=0.6)
        self.ax.set_zlim(-400, 400)
        plt.title('%d epoch' %self.epoch)
        plt.pause(0.01)
        plt.show()