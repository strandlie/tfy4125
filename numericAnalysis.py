import trvalues as tv
import iptrack as it 
import matplotlib.pyplot as plt 
import numpy as np
from math import cos, sin


plt.rcParams.update({'font.size': 15})

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

x_right = 0.62898



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



# COMPARE NUMERICS WITH EXPERIMENTAL.
fullData = np.loadtxt("/Users/hstrandlie/Documents/NTNU/Fysikk/Lab/Lab3/Trackerfiler/fulltrack.txt", skiprows=2)
full_t = fullData[:numOfSteps + 1, :1]
full_x = fullData[:numOfSteps + 1, 1:2]
full_y = fullData[:numOfSteps + 1, 2:]

# COMPUTE VELOCITY
"""
full_v = np.empty(0)
full_v = np.append(full_v, [0])
 
for i in range(len(full_t) - 1):
    
    pos_nPlusOne = np.array([full_x[i+1], full_y[i+1]])
    pos_n = np.array([full_x[i], full_y[i]])

    delta_pos = np.linalg.norm(pos_nPlusOne - pos_n)
    
    delta_t = full_t[i+1] - full_t[i]
    
    full_v = np.append(full_v, (-1)*delta_pos / delta_t)
    
plt.plot(t, v, label="Numerisk")
plt.plot(full_t, full_v, label="Eksperimental")
plt.xlabel("Tid [s]")
plt.ylabel("Hastighet")
plt.legend()
plt.show()
"""

plt.plot(t, y, label="Numerisk")
plt.plot(full_t, full_y, label="Eksperimental")
plt.xlabel("Tid [s]")
plt.ylabel("Y-posisjon [m]")
plt.legend()
plt.show()
