import unittest
import numpy
import irbasis as ir

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Tnl(self):
        for _lambda in [10000.0]:
            prefix =  "basis_f-mp-Lambda"+str(_lambda)+"_np10"
            rf = ir.basis("../irbasis.h5", prefix)
            Tnl = rf.compute_Tnl([0,1,2])


if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
