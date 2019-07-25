from __future__ import print_function
from builtins import range

import unittest
import numpy
import irbasis as ir
import math

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
    x = []
    w = []
    for s in range(ns):
        dx = section_edges[s+1] - section_edges[s]
        x0 = section_edges[s]
        x.extend(((dx/2)*(x_loc+1)+x0).tolist())
        w.extend((w_loc*(dx/2)).tolist())

    return numpy.array(x), numpy.array(w)


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
            for l in range(self._dim):
                self._u_smpl[ix, l] = self._w[ix] * basis.ulx(l, self._x[ix])

    def compute_gl(self, gtau, nl):
        assert nl <= self._dim

        nx = len(self._x)
        gtau_smpl = numpy.zeros((1, nx), dtype=complex)
        for ix in range(nx):
            gtau_smpl[0, ix] = gtau(0.5 * (self._x[ix] + 1) * self._beta)

        return numpy.sqrt(self._beta / 2) * numpy.dot(gtau_smpl[:, :], self._u_smpl[:, 0:nl]).reshape((nl))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_to_Gl(self):
        for _lambda in [10.0, 10000.0]:
            for _statistics in ["f", "b"]:
                beta = 10.0
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                basis = ir.basis("../irbasis.h5", prefix)
                Nl = basis.dim()

                trans = transformer(basis, beta)

                # Trivial test
                gtau = lambda tau: basis.ulx(Nl - 1, 2 * tau / beta - 1)
                gl = trans.compute_gl(gtau, Nl)

                for l in range(Nl - 1):
                    self.assertLessEqual(numpy.abs(gl[l]), 1e-8)
                self.assertLessEqual(numpy.abs(gl[-1] - numpy.sqrt(beta / 2)), 1e-8)

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
