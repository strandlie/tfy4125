import trvalues as tv
import iptrack as it
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

p = it.iptrack("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/bane.txt")
g = 9.81
numOfSteps = 109
h = (1.08)/numOfSteps
height0 = 0.37844
x_left = -0.62898 #prev -0.64808
x_right = 0.71009

def plotTrack():
    x = np.linspace(x_left, x_right, num=numOfSteps)
    y = np.zeros(numOfSteps)
    y = tv.trvalues(p, x)[0]
    
    plt.plot(x,y)
    plt.show()



def getXnPlusOne(xn, alpha, vn, h):
    return xn + (h*vn*cos(alpha))


def getVnPlusOne(vn, alpha, h):
    return vn + (h*g*(5/7)*sin(alpha))

def getYnPlusOne(vn):
    return height0 - (7/10)*((vn**2)/g)


def getTnPlusK(tn, h, k=1):
    return tn + (k*h)


x0 = x_left
y0 = height0
v0 = 0
t0 = 0


# [y,dydx,d2ydx2,alpha,R]
di = {"y": 0, "dydx": 1, "d2ydx2": 2, "alpha": 3, "R": 4}

x_next = x0
y_next = y0
v_next = v0
t_next = t0

x = np.zeros(numOfSteps + 1)
y = np.zeros(numOfSteps + 1)
v = np.zeros(numOfSteps + 1)
t = np.zeros(numOfSteps + 1)

x[0] = x_next
y[0] = y_next
v[0] = v_next
t[0] = t_next


for i in range(1, numOfSteps + 1):
    trv = tv.trvalues(p, x_next)
    alpha = trv[di["alpha"]]
    x_next = getXnPlusOne(x_next, alpha, v_next, h)
    v_next = getVnPlusOne(v_next, alpha, h)
    y_next = trv[di["y"]]
    t_next = t_next + h

    x[i] = x_next
    y[i] = y_next
    v[i] = v_next
    t[i] = t_next


fullData = np.loadtxt("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/fulltrack.txt", skiprows=2)
full_t = fullData[:110, :1]
full_x = fullData[:110, 1:2]
full_y = fullData[:110, 2:]

plt.plot(t, x, label="Numerical")
plt.plot(full_t, (-1)*full_x, label="Experimental")
plt.xlabel("Time")
plt.ylabel("X-position")
plt.legend()
plt.show()


plt.plot(t, y, label="Numerical")
plt.plot(full_t, full_y, label="Experimental")
plt.xlabel("Time")
plt.ylabel("Y-position")
plt.legend()
plt.show()
