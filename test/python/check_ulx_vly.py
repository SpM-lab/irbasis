import unittest
import numpy
import irbasis as ir
import math


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):

        super(TestMethods, self).__init__(*args, **kwargs)

    def test_ulx(self):
        for _lambda in [10.0, 1E+4, 1E+7]:
            prefix = "basis_f-mp-Lambda" + str(_lambda)
            rb = ir.basis("../irbasis.h5", prefix)
            d_ulx_ref = rb._get_d_ulx_ref()
            num_ref_data = d_ulx_ref.shape[0]
            tol = {0: 1e-10, 1: 1e-10, 2: 1e-5}
            for i_ref_data in range(num_ref_data):
                Nl, x, order, ref_val = d_ulx_ref[i_ref_data, :]
                Nl = int(Nl)
                order = int(order)
                val = rb.d_ulx(Nl-1, x, order)
                adiff = abs(ref_val - val)
                rdiff = adiff / ref_val
                print(Nl, x, order, ref_val, val, rdiff)
                self.assertTrue(rdiff < tol[order] or adiff < tol[order])

    def test_vly(self):
        for _lambda in [10.0, 1E+4, 1E+7]:
            prefix = "basis_f-mp-Lambda" + str(_lambda)
            rb = ir.basis("../irbasis.h5", prefix)
            d_vly_ref = rb._get_d_vly_ref()
            num_ref_data = d_vly_ref.shape[0]
            print("Lambda ", _lambda)
            tol = {0: 1e-10, 1: 1e-10, 2: 1e-5}
            for i_ref_data in range(num_ref_data):
                Nl, y, order, ref_val = d_vly_ref[i_ref_data, :]
                Nl = int(Nl)
                order = int(order)
                val = rb.d_vly(Nl-1, y, order)
                adiff = abs(ref_val - val)
                rdiff = adiff / ref_val
                print(Nl, y, order, ref_val, val, rdiff)
                self.assertTrue(rdiff < tol[order] or adiff < tol[order])

    def test_vectorization(self):
        for _lambda in [1E+4]:
            prefix = "basis_f-mp-Lambda" + str(_lambda)
            rb = ir.basis("../irbasis.h5", prefix)

            # re-vectorized functions
            revec_ulx = numpy.vectorize(rb.ulx)
            revec_vly = numpy.vectorize(rb.vly)
            revec_sl = numpy.vectorize(rb.sl)

            # check that those match:
            x = numpy.array([-.3, .2, .5])
            l = numpy.array([1, 3, 10, 15], dtype=int)
            alll = numpy.arange(rb.dim())

            self.assertTrue(numpy.allclose(revec_sl(l), rb.sl(l)))
            self.assertTrue(numpy.allclose(revec_sl(alll), rb.sl()))

            self.assertTrue(numpy.allclose(revec_ulx(l[0], x), rb.ulx(l[0], x)))
            self.assertTrue(numpy.allclose(revec_ulx(l, x[0]), rb.ulx(l, x[0])))
            self.assertTrue(numpy.allclose(revec_ulx(alll, x[0]),
                                           rb.ulx(None, x[0])))
            self.assertTrue(numpy.allclose(revec_ulx(l[:,None], x[None,:]),
                                           rb.ulx(l[:,None], x[None,:])))

            self.assertTrue(numpy.allclose(revec_vly(l[0], x), rb.vly(l[0], x)))
            self.assertTrue(numpy.allclose(revec_vly(l, x[0]), rb.vly(l, x[0])))
            self.assertTrue(numpy.allclose(revec_vly(alll, x[0]),
                                           rb.vly(None, x[0])))
            self.assertTrue(numpy.allclose(revec_vly(l[:,None], x[None,:]),
                                           rb.vly(l[:,None], x[None,:])))



if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
