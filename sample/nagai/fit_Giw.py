from __future__ import print_function

from gf import *
from regression import *
import numpy, scipy
import matplotlib.pyplot as plt

from sklearn import linear_model
import matplotlib.pylab as plt


beta = 10.0
Lambda = 1000.0
pole = 1.0

basis = TwoPointGFBasis('F', Lambda, beta)
Nl = basis.dim()
print("dim ", Nl)

wn = lambda n : (2*n+1)*numpy.pi/beta
Giw = lambda n : 1.0/(1J*wn(n) - pole)
Gtau = lambda tau : -numpy.exp(-pole*tau)/(1+numpy.exp(-beta*pole))

num_data = 200

X = numpy.arange(-int(num_data/2), int(num_data/2))
Y = numpy.array([Giw(n) for n in X])

Gl = fit_Giw(basis, X, Y, solver='svd')

plt.figure(1)
x = numpy.arange(-500, 500, 20)
y = numpy.array([Giw(n) for n in x])
basis.precompute_Tnl(x)
y_fit = numpy.array([basis.evaluate_iwn(Gl, n) for n in x])
plt.plot(x, (numpy.abs(wn(x))+1) * y.imag, marker='x')
plt.plot(x, (numpy.abs(wn(x))+1) * y_fit.imag, marker='+')
plt.savefig("scaled_Giw.pdf")

plt.figure(2)
eps = 1e-8
x = numpy.linspace(eps, beta-eps, 100)
x = numpy.linspace(eps, 0.01*beta-eps, 100)
y_fit = numpy.array([basis.evaluate_tau(Gl, tau) for tau in x])
#plt.plot(x, y_fit.real, marker='+')
plt.plot(x, numpy.abs(y_fit.real-numpy.array([Gtau(tau) for tau in x]) ), marker='+')
plt.savefig("comparison-Gtau.pdf")

plt.figure(3)
plt.plot(numpy.abs(Gl), marker='+')
plt.yscale("log")
#plt.plot(Gl, marker='+')
plt.savefig("Gl.pdf")
