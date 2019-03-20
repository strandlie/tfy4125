import trvalues as tv
import iptrack as it
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

p = it.iptrack("/Users/camillamariedalan/Documents/NTNU/6.Vaar19/Fysikk/rapport_kode/tfy4125/Trackerfiler/bane.txt")
g = 9.81
numOfSteps = 100
h = (1.08)/numOfSteps

def plotTrack():
    x = np.linspace(-0.64808, 0.71009, num=109)
    y = np.zeros(numOfSteps)
    for i in range(numOfSteps):
        y[i] = tv.trvalues(p, x[i])[0]

    plt.plot(x,y)
    plt.show()


def getXnPlusOne(xn, alpha, vn, h):
    return xn + (h*vn*cos(alpha))


def getVnPlusOne(vn, alpha, h):
    return vn + (h*g*(5/7)*sin(alpha))


def getTnPlusK(tn, h, k=1):
    return tn + (k*h)

def getFnPlusOne(fn, alpha, acceleration):
    return fn + g * sin(alpha) - acceleration

def getNnPlusOne(nn, alpha, velocity, radius):
    return ((velocity**2)/radius) + (g * cos(alpha))

# [y,dydx,d2ydx2,alpha,R]
di = {"y": 0, "dydx": 1, "d2ydx2": 2, "alpha": 3, "R": 4}



x0 = -0.585
v0 = 0
t0 = 0

alpha = tv.trvalues(p, x0)[di["alpha"]]
velocity = tv.trvalues(p, x0)[di["dydx"]]
acceleration = tv.trvalues(p, x0)[di["d2ydx2"]]
radius = tv.trvalues(p, x0)[di["R"]]

y0 = tv.trvalues(p, x0)[di["y"]]
f0 = getFnPlusOne(0, alpha, acceleration)
n0 = getNnPlusOne(0, alpha, velocity, radius)


def numericalCalc():
    x = v = t = y = f = n = np.zeros(numOfSteps)

    x_next =  x[0] = x0
    v_next = v[0] = v0
    t_next = t[0] = t0
    y_next = y[0] = y0
    f_next = f[0] = f0
    n_next = n[0] = n0

    for i in range(numOfSteps):
        trv = tv.trvalues(p, x_next)
        y = trv[di["y"]]
        alpha = trv[di["alpha"]]
        velocity = trv[di["dydx"]]
        acceleration = trv[di["d2ydx2"]]
        radius = trv[di["R"]]
        x_next = getXnPlusOne(x_next, alpha, v_next, h)
        v_next = getVnPlusOne(v_next, alpha, h)
        y_next = trv[di["y"]]
        t_next = t_next + h
        f_next = getFnPlusOne(f_next, alpha, acceleration)
        n_next = getNnPlusOne(n_next, alpha, velocity, radius)

        print(y)

        x[i] = x_next
        v[i] = v_next
        t[i] = t_next
        y[i] = y_next
        f[i] = f_next
        n[i] = n_next

    plt.plot(x, f)
    plt.figure()
    plt.plot(x, y)
    #plt.plot(v, f)
    plt.show()

    return x, v, t

def plotFigure(x, y):
    plt.plot(x, y)


if __name__ == '__main__':
    numericalCalc()