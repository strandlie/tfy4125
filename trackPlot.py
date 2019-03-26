import trvalues as tv
import iptrack as it 
import matplotlib.pyplot as plt 
import numpy as np

x_right = 0.629
x_left = -0.598 # Only used to plot the track
p = it.iptrack("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/bane2.txt")

def plotTrack():
    x = np.linspace(x_right, x_left, num=109)
    y = np.zeros(109)
    y = tv.trvalues(p, x)[0]
    plt.plot(x,y)
    plt.show()




if __name__ == '__main__':
    plotTrack()