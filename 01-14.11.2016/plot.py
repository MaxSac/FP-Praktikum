import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
from uncertainties import ufloat
import scipy.constants as c
import math
from lmfit import minimize, Parameter, Model

def stab(r1,r2,L):
    return (1-L/r1)*(1-L/r2)

#
r1 = 0.1; r2 = 0.1;
if r1<r2 :a = r1
elif r1>r2: a = r2 
else: a = 2 * r1
print(a)
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = 0.1 m, r_2 = 0.1 m$')


r1 = 0.5; r2 = 0.5;
if r1<r2 :a = r1
elif r1>r2: a = r2 
else: a = 2*r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = 0.5 m, r_2 = 0.5 m$')

r1 = 1000; r2 = 0.1; L = 0.3
if r1<r2 :a = r1
elif r1>r2: a = r2
else: a = 2*r1
L = np.linspace(0,a,1000)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = 1000 m, r_2 = 0.1 m$')

plt.grid()
plt.legend(loc='best')
plt.savefig('build/Stabilisationsparameter.pdf')
plt.close()
