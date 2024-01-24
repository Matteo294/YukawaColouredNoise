import numpy as np
import scipy.integrate as integrate

masses = [-0.431]

lam = 1.658
g = 0.0
mq = 0.5
cf0 = np.sqrt(2)*np.pi

def tflow_m(m2, t):
    return -lam/4/np.pi * 1/(1+m2*np.exp(-2*t))

def lflow_m(m2, cf):
    return -lam/4/np.pi/cf * 1/(1+m2/cf**2)

def tflow_complete(m2, t):
    return -lam/4/np.pi * 1/(1+m2*np.exp(-2*t)) + 2*g**2/np.pi * 1/(1+mq**2*np.exp(-2*t))

s_05 = []
s_025 = []
s_0125 = []

for m2 in masses:
    z = integrate.odeint(tflow_complete, y0=m2, t=np.log([cf0, cf0/2, cf0/4, cf0/8]))
    s_05.append(z[1][0])
    s_025.append(z[2][0])
    s_0125.append(z[3][0])

    print("flow:", z)

print("s = 1/2:", s_05)
print("s = 1/4:", s_025)
print("s = 1/8:", s_0125)
