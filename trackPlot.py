import trvalues as tv
import iptrack as it
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin

p = it.iptrack("/Users/camillamariedalan/Documents/NTNU/6.Vaar19/Fysikk/rapport_kode/tfy4125/Trackerfiler/bane.txt")
g = 9.81
numOfSteps = 200
h = (1.08)/numOfSteps

# [y,dydx,d2ydx2,alpha,R]
di = {"y": 0, "dydx": 1, "d2ydx2": 2, "alpha": 3, "R": 4}



def getXnPlusOne(xn, alpha, vn):
    return xn + (h*vn*cos(alpha))


def getVnPlusOne(vn, alpha, acceleration):
    return vn + (h*acceleration)
    #return vn + (h * g * (5/7) * sin(alpha))

def getTnPlusK(tn, h, k=1):
    return tn + (k*h)


def getFnPlusOne(fn, alpha, acceleration):
    return (g*sin(alpha) - acceleration)


def getNnPlusOne(nn, alpha, velocity, radius, d2ydx2):
    if d2ydx2 >= 0:
        return ((g * cos(alpha)) + ((velocity ** 2) / radius))
    elif d2ydx2 < 0:
        return ((g * cos(alpha)) - ((velocity ** 2) / radius))

x0 = -0.585

trv = tv.trvalues(p, x0)

alpha = trv[di["alpha"]]
radius = trv[di["R"]]
d2ydx2 = trv[di["d2ydx2"]]


v0 = 0
t0 = 0
a0 = 0
f0 = getFnPlusOne(0, alpha, a0)
n0 = getNnPlusOne(0, alpha, v0, radius, d2ydx2)


def numericalCalc():
    x = np.zeros(numOfSteps)
    v = np.zeros(numOfSteps)
    t = np.zeros(numOfSteps)
    f = np.zeros(numOfSteps)
    n = np.zeros(numOfSteps)
    a = np.zeros(numOfSteps)
    y = np.zeros(numOfSteps)

    x_next = x0
    v_next = v0
    t_next = t0
    f_next = f0
    n_next = n0
    a_next = a0

    trvalues = tv.trvalues(p, x0)

    x[0] = x_next
    v[0] = v_next
    t[0] = t_next
    f[0] = f_next
    n[0] = n_next
    a[0] = a0
    y[0] = trvalues[di["y"]]

    for i in range(numOfSteps):
        trv = tv.trvalues(p, x_next)
        d2ydx2 = trv[di["d2ydx2"]]
        alpha = trv[di["alpha"]]
        radius = trv[di["R"]]


        a_next = g*(5/7)*sin(alpha)
        v_next = getVnPlusOne(v_next, alpha, a_next)
        x_next = getXnPlusOne(x_next, alpha, v_next)

        t_next = t_next + h
        f_next = getFnPlusOne(f_next, alpha, a_next)
        n_next = getNnPlusOne(n_next, alpha, v_next, radius, d2ydx2)

        y[i] = trv[di["y"]]
        x[i] = x_next
        v[i] = v_next
        t[i] = t_next
        f[i] = f_next
        n[i] = n_next
        a[i] = a_next


    plt.plot(x, f)
    plt.xlabel('X-posisjon')
    plt.ylabel('Friksjon per masse')

    plt.figure()
    plt.plot(x, n)
    plt.xlabel('X-posisjon')
    plt.ylabel('Normalkraft per masse')

    plt.figure()
    plt.plot(x, y)
    plt.xlabel('X-posisjon')
    plt.ylabel('Y-posisjon')
    plt.title('Bane')

    plt.figure()
    plt.plot(x, v)
    plt.xlabel('X-posisjon')
    plt.ylabel('Fart')

    plt.figure()
    plt.plot(x, a)
    plt.xlabel('X-posisjon')
    plt.ylabel('Akselerasjon')

    plt.show()



if __name__ == '__main__':
    numericalCalc()