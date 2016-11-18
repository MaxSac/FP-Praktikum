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

r1 = 1000000; r2 = 1.4;
if r1<r2 : a = r1
elif r1>r2: a = r2
else: a = 2 * r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label=r'$r_1 = \, \text{flat} , r_2 = 1.4 \, \text{m} $')


r1 = 1.4; r2 = 1.4;
if r1<r2 : a = r1
elif r1>r2: a = r2
else: a = 2*r1
L = np.linspace(0,a,100)
plt.plot(L ,stab(r1, r2, L),label=r'$r_1 = \, \text{flat}, r_2 = 1.4 \, \text{m}$')

plt.grid()
plt.xlim(xmin=0, xmax=2.8)
plt.ylim(ymax=1)
plt.xlabel('Resonatorlänge / m')
plt.ylabel('Stabilitätsparameter')
plt.legend(loc='best')
plt.savefig('build/Stabilisationsparameter.pdf')
plt.close()

#Intensität in Abhängikeit der Polarisation
winkel, intensi = np.loadtxt('Polarisation.txt', unpack=True)
winkel = np.radians(winkel)
def fsin(x, a, b, c, d):
    return (np.sin(b*x +c)**2)*a +d 

params, cov = curve_fit(fsin, winkel, intensi,
        bounds=([150,0.5,0,0],[250,1.5,2*np.pi,50]))
linw = np.linspace(0,max(winkel),100)
plt.plot(winkel, intensi, 'x', label='Messwerte')
plt.plot(linw , fsin(linw, *params), label='Fit')
plt.legend(loc='best')
plt.savefig('build/Polar.pdf')
plt.close()

#Intensität der verschiedenen TEM Moden
A, TEM00, TEM10 = np.loadtxt('TEM-Moden.txt', unpack=True)

def gausian(x, a, v, my):
    return a*np.exp(-(x+my)**2/(v**2))
params, cov = curve_fit(gausian, A, TEM00, bounds=([3,5,-20],[6,10,-10]))

def gausian2( x, a, v, my):
    return a*8*(x-my)**2/v**2*np.exp(-0.5*(x-my)**2/v**2)
params2, cov2 = curve_fit(gausian2, A, TEM10,bounds=([0,0,12],[1,10,20])) 

x = np.linspace(0,max(A+1),100)
plt.plot(x, gausian(x, *params))
plt.plot(A, TEM00, 'x', label='TEM00')
plt.xlim(xmin=0, xmax=max(A+1))
plt.legend(loc='best')
plt.savefig('build/TEM00.pdf')
plt.close()

plt.plot(x, gausian2(x, *params2))
plt.plot(A, TEM10, 'x', label='TEM10')
plt.xlim(xmin=0, xmax=max(A+1))
plt.legend(loc='best')
plt.savefig('build/TEM10.pdf')
plt.close()
