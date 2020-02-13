from __future__ import print_function
from builtins import range

import unittest
import numpy
import h5py
import irbasis as ir
import math
from itertools import product

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_unl(self):
        """
        Consider a pole at omega=pole. Compare analytic results of G(iwn) and numerical results computed by using unl.
        """
        for _lambda in [10.0, 1E+4, 1E+7]:
            for _statistics, pole in product(["f", "b"], [1.0, 0.1]):
                print("lambda = %d, stat = %s, y = %g" %
                      (_lambda, repr(_statistics), pole))
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                basis = ir.basis("../irbasis.h5", prefix)
                dim = basis.dim()

                wmax = 1.0
                beta = _lambda/wmax

                if _statistics == 'f':
                    rho_l = numpy.sqrt(1/wmax)* numpy.array([basis.vly(l, pole/wmax) for l in range(dim)])
                    Sl = numpy.sqrt(0.5 * beta * wmax) * numpy.array([basis.sl(l) for l in range(dim)])
                    stat_shift = 1
                else:
                    rho_l = numpy.sqrt(1/wmax)* numpy.array([basis.vly(l, pole/wmax) for l in range(dim)])/pole
                    Sl = numpy.sqrt(0.5 * beta * wmax**3) * numpy.array([basis.sl(l) for l in range(dim)])
                    stat_shift = 0
                gl = - Sl * rho_l

                def G(n):
                    wn = (2*n+stat_shift)*numpy.pi/beta
                    z = 1J * wn
                    return 1/(z - pole)

                # Compute G(iwn) using unl
                n_plt = numpy.array([-1, 0, 1, 1E+1, 1E+2, 1E+3, 1E+4,
                                     1E+5, 1E+6, 1E+7, 1E+8, 1E+9, 1E+10, 1E+14],
                                    dtype=int)
                Uwnl_plt =  numpy.sqrt(beta) * basis.compute_unl(n_plt)
                Giwn_t = numpy.dot(Uwnl_plt, gl)

                # Compute G(iwn) from analytic expression
                Giwn_ref = numpy.array([G(n) for n in n_plt])

                magnitude = numpy.abs(Giwn_ref).max()
                diff = numpy.abs(Giwn_t - Giwn_ref)
                reldiff = diff/numpy.abs(Giwn_ref)

                # Absolute error must be smaller than 1e-12
                print ("max. absdiff = %.4g, rel = %.4g" %
                       (diff.max()/magnitude, reldiff.max()))
                self.assertLessEqual((diff/magnitude).max(), 5e-13)

                # Relative error must be smaller than 1e-12
                #self.assertLessEqual(numpy.amax(numpy.abs(diff/Giwn_ref)), 1e-12)


if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
