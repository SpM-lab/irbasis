import unittest
import numpy
from basis import *
import scipy.integrate as integrate

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        max_dim = 1000
        self.b = basis_f(100.0, max_dim)
        self.b.compute_Tnl(numpy.array([0, 1, 2]))
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_norm(self):
        for l in range(self.b.dim()):
            self.assertTrue(numpy.abs(integrate.quad(lambda x: self.b.ulx(l,x)**2, -1.0, 1.0, epsabs=1e-6, limit=400)[0]-1) < 1e-8)

    def test_small_lambda_f(self):
        for Lambda in [0.1, 1.0]:
            b = basis_f(Lambda)

    def test_small_lambda_b(self):
        for Lambda in [0.1, 1.0]:
            b = basis_b(Lambda)

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
