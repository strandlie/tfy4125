import trvalues as tv
import trackPlot
import iptrack as it
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin


p = it.iptrack("/Users/camillamariedalan/Documents/NTNU/6.Vaar19/Fysikk/rapport_kode/tfy4125/Trackerfiler/bane.txt")
g = 9.81
numOfSteps = 100
h = (1.08)/numOfSteps

def plotTrack():
    x = np.linspace(-0.64808, 0.71009, num=numOfSteps)
    #x = np.linspace(-0.6, 0.6, num=numOfSteps)
    values = tv.trvalues(p, x)
    y, alpha, radius = values[0], values[3], values[4]

    plt.plot(x, radius)
    plt.figure()
    plt.plot(x, alpha)
    plt.figure()
    plt.plot(x,y)
    plt.show()


def plotFrictionPerMass():
    friction = g * sin(x) - dydx



def plotNormalForcePerMass():
    pass

if __name__ == '__main__':
    x, v, t = trackPlot.numericalCalc()
    print(x, v, t)