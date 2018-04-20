import unittest
import numpy
import irbasis as ir
import math

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_small_lambda_f(self):
        for _lambda in [10.0, 10000.0]:
            prefix =  "basis_f-mp-Lambda"+str(_lambda)
            rb = ir.basis("../irbasis.h5", prefix)
            check_data_ulx = rb.check_ulx()
            for _data in check_data_ulx:
                self.assertLessEqual( _data[2], math.pow(10.0 , -8))

            check_data_vly = rb.check_vly()
            for _data in check_data_vly:
                self.assertLessEqual( _data[2], math.pow(10.0 , -8))

                
    def test_small_lambda_b(self):
        for _lambda in [10.0, 10000.0]:
            prefix =  "basis_b-mp-Lambda"+str(_lambda)
            rb = ir.basis("../irbasis.h5", prefix)
            check_data_ulx = rb.check_ulx()
            for _data in check_data_ulx:
                self.assertLessEqual( _data[2], math.pow(10.0 , -8))

            check_data_vly = rb.check_vly()
            for _data in check_data_vly:
                self.assertLessEqual( _data[2], math.pow(10.0 , -8))
                

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
