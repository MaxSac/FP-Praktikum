import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks_cwt
from uncertainties import ufloat
import scipy.constants as c
from lmfit import minimize, Parameter, Model

x = np.linspace(0, 2*np.pi)
y = np.sin(x)

plt.plot(x,y)
plt.savefig('Bilder/test.pdf')
