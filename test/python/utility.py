from __future__ import print_function
from builtins import range

import unittest
import numpy
import irbasis as ir
import math

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_to_Gl(self):
        for _lambda in [10.0, 10000.0]:
            for _statistics in ["f", "b"]:
                beta = 10.0
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                basis = ir.basis("../irbasis.h5", prefix + "_np8")
                Nl = basis.dim()

                trans = ir.transformer(basis, beta)

                # Trivial test
                gtau = lambda tau: basis.ulx(Nl - 1, 2 * tau / beta - 1)
                gl = trans.compute_gl(gtau, Nl)

                for l in range(Nl - 1):
                    self.assertLessEqual(numpy.abs(gl[l]), 1e-8)
                self.assertLessEqual(numpy.abs(gl[-1] - numpy.sqrt(beta / 2)), 1e-8)

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
