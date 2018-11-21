import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from uncertainties import ufloat

v1, u1 = np.genfromtxt("integrator.csv", delimiter=';', unpack=True)
v2, u2 = np.genfromtxt("differentiator.csv", delimiter=';', unpack=True)


def f(x, a, b):
    y=a*x**b
    return y

popt1, pcov1 = curve_fit(f, v1[:12], u1[:12])
a1=ufloat(popt1[0], np.sqrt(np.diag(pcov1))[0])
b1=ufloat(popt1[1], np.sqrt(np.diag(pcov1))[1])
x1=np.linspace(40, 700, 1000)

popt2, pcov2 = curve_fit(f, v2, u2)
a2=ufloat(popt2[0], np.sqrt(np.diag(pcov2))[0])
b2=ufloat(popt2[1], np.sqrt(np.diag(pcov2))[1])
x2=np.linspace(40, 1000, 1000)

print("Integrator", a1, b1)
print("Differentiator", a2, b2)

plt.figure()
plt.plot(v1, u1, "x", label="Data")
plt.plot(x1, f(x1, *popt1), label="Fit")
plt.xlabel(r"$\nu$ / Hz")
plt.ylabel(r"$U_A$ / V")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("plot4.pdf", bbox_inches="tight")

plt.figure()
plt.plot(v2, u2, "x", label="Data")
plt.plot(x2, f(x2, *popt2), label="Fit")
plt.xlabel(r"$\nu$ / Hz")
plt.ylabel(r"$U_A$ / V")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("plot5.pdf", bbox_inches="tight")

plt.show()
