import numpy as np
import matplotlib.pyplot as plt

def getHeight(t, h0=0.37844, gamma=0.0773332328106):
    return h0*np.exp(-gamma*t)


data = np.loadtxt("/Users/camillamariedalan/Documents/NTNU/6.Vaar19/Fysikk/rapport_kode/tfy4125/Trackerfiler/fulltrack.txt", skiprows=2)

t = data[:, :1]     # All rows, first column
x = data[:, 1:2]    # A.pyplotll rows, second column 
y = data[:, 2:]     # All rows, third column


plt.plot(t, y)
plt.xlabel('Tid')
plt.ylabel('Høyde')
#plt.plot(t, getHeight(t))
plt.show()


