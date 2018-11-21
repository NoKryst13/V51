import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from uncertainties import ufloat

# hier Messdaten als csv auslesen
vv1,u11,ua1 = np.genfromtxt("a1.csv", delimiter=';', unpack=True)
vv2,u12,ua2 = np.genfromtxt("a2.csv", delimiter=';', unpack=True)
vv3,u13,ua3 = np.genfromtxt("a3.csv", delimiter=';', unpack=True)

v1, v2, v3 = vv1*1000, vv2*1000, vv3*1000
#################################################################################################################
# Konstanten, bekannte Parameter
V1=ua1/(u11*10**(-3))
V2=ua2/(u12*10**(-3))
V3=ua3/(u13*10**(-3))

#################################################################################################################
# Neue csv mit berechneten Listen
# with open('test.csv', 'w') as f:    # neue csv schreiben aus bestehenden Listen -> Spalten
#     writer = csv.writer(f)
#     f.write("Position;Intensitat;Position2;Intensitat2\n") # Den Spalten Überschriften geben, \n für neue Zeile
#     writer.writerows(zip(liste1,liste2,...))

#################################################################################################################
# Fit a1
def f(x, a, b):
    y=a*x**b
    return y

popt1, pcov1 = curve_fit(f, v1[10:31], V1[10:31])
a1=ufloat(popt1[0], np.sqrt(np.diag(pcov1))[0])
b1=ufloat(popt1[1], np.sqrt(np.diag(pcov1))[1])
x1 = np.linspace(50000, 1000000, 10000)
vg1=((np.sqrt(100/2)/a1)**(1/b1))

print("Parameter Gerade1 ( y = ax^b ):", a1, b1)
print("Grenzfrequenz vg1:", vg1)
print("Verstärkung-Bandbreite-Produkt (1, V=10):", vg1*10)


popt3, pcov3 = curve_fit(f, v3[6:], V3[6:])
a3=ufloat(popt3[0], np.sqrt(np.diag(pcov3))[0])
b3=ufloat(popt3[1], np.sqrt(np.diag(pcov3))[1])
vg3=((2/0.023)*(np.sqrt(1/2)/a3))**(1/b3)
x3 = np.linspace(9500, 1000000, 10000)

# x3x = np.linspace(4330, 13000, 1000)
# lin1 = np.array([[1, 1/np.log(v3[2])], [1, 1/np.log(v3[3])]])
# lin2 = np.array([np.log(V3[2])/np.log(v3[2]), np.log(V3[3])/np.log(v3[3])])
# b3x, a3x = np.linalg.solve(lin1, lin2)
# vg3x=((np.sqrt(10000/2)/(a3x*100000))**(1/b3x))
# print("solve", a3x*100000, b3x)

print("Parameter Gerade3 ( y = ax^b ):", a3, b3)
print("Grenzfrequenz vg3:", vg3)
print("Verstärkung-Bandbreite-Produkt (3, V=100):", vg3*(2/0.023))
print("Prozentuale Abweichung VBP:", 100*(vg1*10-vg3*(2/0.023))/(vg1*10))
print("Grenzfrequenz vg3 alternativ:", 10800, 10800*(2/0.023))
print("Prozentuale Abweichung VBP alternativ:", 100*(vg1*10-10800*(2/0.023))/(vg1*10))
################################################################################################################# Plots
plt.figure()
plt.plot(v1, V1, "x", label="Data", ms=10)
plt.plot(x1, f(x1, *popt1), label="Fit")
plt.ylabel("V'")
plt.xlabel(r"$\nu$ / Hz")
plt.xscale("log")
plt.yscale("log")
plt.hlines((100/2)**(1/2),0,110000, colors='r', linestyle="dotted")
plt.text(783.67, 7.065,r"$\frac{V´}{\sqrt{2}}$", fontsize=12, color="r")
plt.vlines(79000, 0, 10, linestyle="dotted", color="r")
plt.text(68000, 0.65 ,r"$\nu_g$", fontsize=10, color="r")
plt.grid()
plt.legend()
plt.savefig("plot1.pdf", bbox_inches="tight")

plt.figure()
plt.plot(v2, V2, "x", label="Data", ms=10)
plt.ylabel("V'")
plt.xlabel(r"$\nu$ / Hz")
plt.xscale("log")
plt.yscale("log")
plt.grid()
plt.legend()
plt.savefig("plot2.pdf", bbox_inches="tight")

plt.figure()
plt.plot(v3, V3, "x", label="Data", ms=10)
plt.plot(x3, f(x3, *popt3), label="Fit")
plt.plot([v3[3], v3[4]],[V3[3], V3[4]], label="Alternative Fit")
plt.ylabel("V'")
plt.xlabel(r"$\nu$ / Hz")
plt.xscale("log")
plt.yscale("log")
plt.hlines((2/0.023)*(1/2)**(1/2),0,110000, colors='r', linestyle="dotted")
plt.text(110, (2/0.023)*(1/2)**(1/2)-4,r"$\frac{V´}{\sqrt{2}}$", fontsize=12, color="r")
plt.vlines(13700, 0, (2/0.023)*(1/2)**(1/2), linestyle="dotted", color="r")
plt.vlines(10800, 0, (2/0.023)*(1/2)**(1/2), linestyle="dotted", color="g")
plt.text(12700, 0.7 ,r"$\nu_g'$", fontsize=10, color="r")
plt.grid()
plt.legend()
plt.savefig("plot3.pdf", bbox_inches="tight")

plt.show()
