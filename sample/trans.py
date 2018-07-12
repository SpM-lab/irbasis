from __future__ import print_function
#from builtins import range

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import irbasis
plt.rc('text', usetex=True)
def _composite_leggauss(deg, section_edges):
    """
    Composite Gauss-Legendre quadrature.

    :param deg: Number of sample points and weights. It must be >= 1.
    :param section_edges: array_like
                          1-D array of the two end points of the integral interval
                          and breaking points in ascending order.
    :return ndarray, ndarray: sampling points and weights
    """
    x_loc, w_loc = numpy.polynomial.legendre.leggauss(deg)

    ns = len(section_edges)-1
    x = numpy.zeros((ns, deg))
    w = numpy.zeros((ns, deg))
    for s in range(ns):
        dx = section_edges[s+1] - section_edges[s]
        x0 = section_edges[s]
        x[s, :] = (dx/2)*(x_loc+1) + x0
        w[s, :] = w_loc*(dx/2)
    return x.reshape((ns*deg)), w.reshape((ns*deg))

class transformer(object):
    def __init__(self, basis, beta):
        section_edges_positive_half = numpy.array(basis.section_edges_x)
        section_edges = numpy.setxor1d(section_edges_positive_half, -section_edges_positive_half)
        self._dim = basis.dim()
        self._beta = beta
        self._x, self._w = _composite_leggauss(24, section_edges)

        nx = len(self._x)
        self._u_smpl = numpy.zeros((nx, self._dim))
        for ix in range(nx):
            self._u_smpl[ix, :] = self._w[ix] * basis.ulx_all_l(self._x[ix])

    def compute_gl(self, gtau, nl):
        assert nl <= self._dim

        nx = len(self._x)
        gtau_smpl = numpy.zeros((1, nx), dtype=complex)
        for ix in range(nx):
            gtau_smpl[0, ix] = gtau(0.5 * (self._x[ix] + 1) * self._beta)

        return numpy.sqrt(self._beta / 2) * numpy.dot(gtau_smpl[:, :], self._u_smpl[:, 0:nl]).reshape((nl))

if __name__ == '__main__':
    # Fermion, Lambda = 1000.0
    beta = 100.0
    Lambda = 1000.0
    #for Lambda in [100.0,1000.0,10000.0]:
    basis = irbasis.load('F',  Lambda)
    Nl = basis.dim()
    linst = ['solid','dashed','dashdot','dotted']
    trans = transformer(basis, beta)

    # Transform the highest-order basis function to IR basis (trivial case)
    gtau = lambda tau: basis.ulx(Nl - 1, 2 * tau / beta - 1)
    gl = trans.compute_gl(gtau, Nl)

    #assert numpy.all(numpy.abs(gl[:-1]) < 1e-8)
    #assert numpy.abs(gl[-1] - numpy.sqrt(beta / 2)) < 1e-8
    Tn = basis.compute_unl([0])
    #print(Tn)
    Tnr = Tn.real
    #print(Tnr)
    Tnre = Tnr.size
    mi = numpy.linspace(0,Tnre,Tnre)
    #mi = numpy.linspace(-1,1,10000)
    print(gl)
    #print(gtau(1))
    #for l in [0]:
        #plt.plot(mi,numpy.abs(Tnr[0][:]),linestyle = linst[1])
        #plt.plot(mi,numpy.array([numpy.abs(basis.ulx(l,x)) for x in numpy.linspace(-1,1,10000)]),linestyle = linst[1],label = "$l = {}$".format(l))
        
    #for l in range(gl.size):
    #    goftau = gl[l]*gtau(tau)
    #plt.xlim(-1.1,1.1)
    #plt.ylim(0.001,10.1)
    #plt.xscale("log")
    
    plt.yscale("log")
    plt.xlabel(r'$x$',fontsize = 18)
    plt.tick_params(labelsize=21)
    #plt.plot(mi,Tn.imag[0,:])
    plt.ylabel(r'${}[unl^{}]$'.format(r"\rm{Re}",r"{\rm{F}}"),fontsize = 21)
    plt.legend(frameon=False,fontsize = 21)
    plt.tight_layout()
    #plt.savefig('u'+'.pdf')  
    plt.show()
   #plt.plot(mi,numpy.array([basis.ulx(0,x) for x in numpy.linspace(-1,1,1000)]))
    
    
    

