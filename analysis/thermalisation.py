from matplotlib import pyplot as plt
from pandas import read_csv
import numpy as np

N = 100
Nskip = 20

data = read_csv("traces.csv")
data2 = read_csv("traces2.csv")
data3 = read_csv("traces3.csv")

mean1 = np.mean(data['sigma'][Nskip:])
mean2 = np.mean(data2['sigma'][Nskip:])
mean3 = np.mean(data3['sigma'][Nskip:])

xpoints = range(N)

plt.plot(xpoints, data['sigma'][:N], label="s=1", color='blue')
plt.plot([0, N-1], [mean1, mean1], color='blue')
plt.plot(xpoints, data2['sigma'][:N], label="s=1/2", color='green')
plt.plot([0, N-1], [mean2, mean2], color='green')
plt.plot(xpoints, data3['sigma'][:N], label="s=0", color='red')
plt.plot([0, N-1], [mean3, mean3], color='red')
plt.grid()
plt.xlabel('Langevin steps')
plt.ylabel(r'$\phi$')
plt.legend()
plt.tight_layout()
plt.savefig('thermalisation.pdf')
plt.close()

plt.hist(data['sigma'][Nskip:], bins=50, density=True, color='blue', label='s=1')
plt.hist(data2['sigma'][Nskip:], bins=50, density=True, color='green', alpha=0.7, label='s=1/2')
plt.hist(data3['sigma'][Nskip:], bins=50, density=True, color='red', alpha=0.3, label='s=1/2')
plt.legend()
plt.savefig('hist.pdf')
plt.close()
