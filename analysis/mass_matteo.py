from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit as fit
import numpy as np
import sys

'''m0 = 0.2
yukawa_coupling = 1.0
sigma = 0.2
pi = [0.3, 0.4, 0.1]'''

mq0 = 0.3
yukawa_coupling = 1.0
sigma = 0.10064470516582914
pi = [-0.00042856927638190923, 0.0012371204007035174, -0.0006443482380904521]

plot = True

def expectedM(m0, g, sigma, pi):
    r2 = sigma**2 + pi[0]**2 + pi[1]**2 + pi[2]**2
    denom = 2*(g*sigma + m0 + 1)
    sqrroot = np.sqrt((g**2*r2 + 2*m0*(g*sigma + 1) + 2*g*sigma + m0**2 + 2)**2 - 4*(g*sigma+m0+1)**2)
    num = -sqrroot + g**2*r2 + 2*g*m0*sigma + 2*g*sigma + m0**2 + 2*m0 + 2
    return -np.log(num/denom)

def fitfuncSinh(x, m_re, A):
    return A * np.sinh(m_re*(Nt/2-x))

def fitToSinh(ydata, startidx, endidx, plot=False):
    yvals = ydata[startidx:endidx]
    xvals = np.array(range(startidx, endidx))

    fitparams = fit(fitfuncSinh, xvals, yvals, p0=[1e-2, np.log(1+mq0+yukawa_coupling*sigma)])
    #print("Mass: ", abs(fitparams[0][0]))
        
    return fitparams[0]

    '''plt.plot(xvals, fitfuncSinh(xvals, fitparams[0][0], fitparams[0][1]), label="fit")
    plt.plot(xvals, yvals, '.', markersize=6, label="data")
    plt.legend()
    plt.title("Temporal correlator m0=" + str(m0) + " g=" + str(g) + r" $\phi=[$" + str(sigma) + ", " + str(pi[0]) + ", " + str(pi[1]) + ", " + str(pi[2]) + "]")
    plt.xlabel("t")
    plt.show()'''

def get_phys_quark_mass_via_timeslices(Sq_t,volume):
    global Nt 
    Nt = int(volume[0])
    corr = np.average(Sq_t, axis=0)
    try:
    	val, err = fitToSinh(corr, 1, Nt, plot=False)
    except:
    	val = 0
    	err = 0
    return [val, err]
  
'''
data = read_csv(sys.argv[1])
Nt = int(sys.argv[2])  

timeslices = data['corr'].to_numpy(np.dtype('f8')).reshape((-1, Nt))


corr = np.average(timeslices, axis=0)
print(corr)

f = open("averaged.csv", "w")
for i in range(len(corr)-1):
    f.write(str(i+1) + "," + str(corr[i+1]) + "\n")
f.close()

fitparams = fitToSinh(corr, 1, Nt, plot)

if plot:
    plt.plot(range(2,Nt-1), corr[2:Nt-1], label="data")
    xvals = np.linspace(0, Nt, 1000)
    plt.plot(xvals, fitfuncSinh(xvals, fitparams[0], fitparams[1]), label='fit')
    plt.xlabel(r'$N_t$')
    plt.title("Correlator")
    plt.tight_layout()
    plt.savefig("mass.pdf")
    plt.close()

print("Params: ")
print("mq0:", mq0)
print("g:", yukawa_coupling)
print("Number of data points:", timeslices.shape[0])
print("Nt:", timeslices.shape[1])
print("Measured mass:", fitparams[0])
print("Expected mass with phi =", np.concatenate(([sigma], pi)), ":", expectedM(mq0, yukawa_coupling, sigma, pi))
'''
