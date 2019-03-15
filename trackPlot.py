import trvalues as tv
import iptrack as it
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

p = it.iptrack("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/bane.txt")
g = 9.81
numOfSteps = 109
h = (1.08)/numOfSteps

def plotTrack():
    x = np.linspace(-0.64808, 0.71009, num=109)
    y = np.zeros(109)
    for i in range(109):
        y = tv.trvalues(p, x)[0]


    plt.plot(x,y)
    plt.show()



def getXnPlusOne(xn, alpha, vn, h):
    return xn + (h*vn*cos(alpha))


def getVnPlusOne(vn, alpha, h):
    return vn + (h*5*sin(alpha)/7)


def getTnPlusK(tn, h, k=1):
    return tn + (k*h)


x0 = -0.585
v0 = 0
t0 = 0

# [y,dydx,d2ydx2,alpha,R]
di = {"y": 0, "dydx": 1, "d2ydx2": 2, "alpha": 3, "R": 4}

x_next = x0
v_next = v0
t_next = t0

x = np.zeros(numOfSteps)
v = np.zeros(numOfSteps)
t = np.zeros(numOfSteps)

x[0] = x_next
v[0] = v_next
t[0] = t_next

for i in range(numOfSteps):
    trv = tv.trvalues(p, x_next)
    alpha = trv[di["alpha"]]
    x_next = getXnPlusOne(x_next, alpha, v_next, h)
    v_next = getVnPlusOne(v_next, alpha, h)
    t_next = t_next + h

    x[i] = x_next
    v[i] = v_next
    t[i] = t_next


plt.plot(x, v)
plt.show()
