import trvalues as tv
import iptrack as it 
import matplotlib.pyplot as plt 
import numpy as np
from math import cos, sin


def getXnPlusOne(xn, alpha, vn, h):
    return xn + (h*vn*cos(alpha))


def getVnPlusOne(vn, alpha, h):
    return vn + (h*g*(5.0/7.0)*sin(alpha))


# [y,dydx,d2ydx2,alpha,R]
di = {"y": 0, "dydx": 1, "d2ydx2": 2, "alpha": 3, "R": 4}

# CONSTANTS
p = it.iptrack("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/bane2.txt")
g = 9.81
numOfSteps = 98

t0 = 0
t_end = float(numOfSteps) / 100
h = (t_end - t0)/numOfSteps

x_right = 0.629



# Initial values
x0 = x_right
v0 = 0
y0 = tv.trvalues(p, x0)[di["y"]]


x = np.zeros(numOfSteps + 1)
v = np.zeros(numOfSteps + 1)
y = np.zeros(numOfSteps + 1)
t = np.zeros(numOfSteps + 1)

x[0] = x0
v[0] = v0
y[0] = y0
t[0] = t0

t_old = t0
x_old = x0
v_old = v0

for i in range(numOfSteps):
    trv = tv.trvalues(p, x_old)
    
    alpha = trv[di["alpha"]]
    y_new = trv[di["y"]]
    
    x_new = getXnPlusOne(x_old, alpha, v_old, h)
    v_new = getVnPlusOne(v_old, alpha, h)
    
    x[i+1] = x_new
    v[i+1] = v_new
    y[i+1] = y_new
    t[i+1] = t_old + h
    
    t_old = t_old + h
    x_old = x_new
    v_old = v_new


"""
plt.plot(t, x)
plt.xlabel("Tid")
plt.ylabel("Hastighet")
plt.show()


# COMPARE NUMERICS WITH EXPERIMENTAL. TODO: Has a bug, in that it doesn't start at the same y-value. 
"""
fullData = np.loadtxt("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/fulltrack.txt", skiprows=2)
full_t = fullData[:numOfSteps + 1, :1]
full_x = fullData[:numOfSteps + 1, 1:2]
full_y = fullData[:numOfSteps + 1, 2:]

for i in range(len(y)):
    txy = "t[" + str(i) + "]:" + str(t[i]) + "\t\t x[" + str(i) + "]: " + str(x[i]) + "\t\t y[" + str(i) + "]: " + str(y[i])
    full_txy = "\t\t full_t[" + str(i) + "]:" + str(full_t[i]) + "\t\t full_x[" + str(i) + "]: " + str(full_x[i]) + "\t\t full_y[" + str(i) + "]: " + str(full_y[i])
    
    print(txy + full_txy)

plt.plot(t, y, label="Numerisk")
plt.plot(full_t, full_y, label="Eksperimental")
plt.xlabel("Tid")
plt.ylabel("Y-posisjon")
plt.legend()
plt.show()