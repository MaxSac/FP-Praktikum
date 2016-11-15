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

r1 = 1000000; r2 = 1000000;
if r1<r2 : a = r1
elif r1>r2: a = r2
else: a = 2 * r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = flat , r_2 = flat $')


r1 = 100000000; r2 = 1.4;
if r1<r2 : a = r1
elif r1>r2: a = r2
else: a = 2*r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = flat, r_2 = 1.4 m$')


r1 = 1.5; r2 = 1.0;
if r1<r2 : a = r1
elif r1>r2: a = r2
else: a = 2*r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label='$r_1 = 2.0, r_2 = 1.0 m$')

plt.grid()
plt.xlim(xmin=0, xmax=2)
plt.ylim(ymax=1.1)
plt.xlabel('Resonatorlänge / m')
plt.ylabel('Stabilitätsparameter')
plt.legend(loc='best')
plt.savefig('build/Stabilisationsparameter.pdf')
plt.close()

winkel, intensi = np.loadtxt('Polarisation.txt', unpack=True)
winkel = np.radians(winkel)
def fsin(x, a, b, c, d):
    return np.sin(b*x +c)*a +d 

params, cov = curve_fit(fsin, winkel, intensi, bounds=([80,2,1.3,70],[150,4,10,150]))
plt.plot(winkel, intensi, 'x')
plt.plot(winkel, fsin(winkel, *params))
plt.savefig('build/Polar.pdf')
plt.close()

A, TEM00, TEM10 = np.loadtxt('TEM-Moden.txt', unpack=True)
def gausian(x, a, v, my):
    return a*np.exp(-(x+my)**2/(v**2))
params, cov = curve_fit(gausian, A, TEM00, bounds=([3,5,-20],[6,10,-10]))
print(params)
x = np.linspace(0,35,100)
plt.plot(x, gausian(x, *params))
plt.plot(A, TEM00, 'x', label='TEM00')
plt.plot(A, TEM10, 'x', label='TEM10')
plt.legend(loc='best')
plt.savefig('build/Modenblende.pdf')
plt.close()
