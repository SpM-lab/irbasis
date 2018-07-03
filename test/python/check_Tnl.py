from __future__ import print_function
from builtins import range

import unittest
import numpy
import h5py
import irbasis as ir
import math

class refdata(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'].value
            self._dim = f[prefix+'/info/dim'].value

            self._unl_odd_ref = f[prefix+'/data/lodd/Tnl'].value
            self._unl_odd_ref_max = f[prefix+'/data/lodd/Tnlmax'].value
            self._unl_odd_l = f[prefix+'/data/lodd/l'].value
            
            self._unl_even_ref = f[prefix+'/data/leven/Tnl'].value
            self._unl_even_ref_max = f[prefix+'/data/leven/Tnlmax'].value
            self._unl_even_l = f[prefix+'/data/leven/l'].value
            
            
    def check_data(self, basis, statistics):
            #Check odd-l
            l = self._unl_odd_l 
            unl = basis.compute_unl(numpy.array(self._unl_odd_ref[:, 0], dtype=int))[:, l]
            dunl = abs(unl- self._unl_odd_ref[:,1])/self._unl_odd_ref_max

            if statistics == "f":
                unl_limit = -(basis.d_ulx(l, 1, 1)+basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            else:
                unl_limit = -(basis.ulx(l, 1)-basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            nvec = self._unl_odd_ref[:,0]
            if statistics == "f":
                unl_coeff = nvec*nvec*unl.real
            else:
                unl_coeff = nvec*unl.imag
            dunl_coeff= abs(unl_limit-unl_coeff[len(unl_coeff)-1]) \
                    if abs(unl_limit) < 1e-12 \
                    else abs(unl_limit-unl_coeff[len(unl_coeff)-1])/abs(unl_limit)
            dunl_limit =dunl_coeff

            #Check even-l
            l = self._unl_even_l 
            unl = basis.compute_unl(numpy.array(self._unl_even_ref[:, 0], dtype=int))[:, l]
            dunl = numpy.append(dunl, abs(unl- self._unl_even_ref[:,1])/self._unl_even_ref_max)
            
            if statistics == "f":
                unl_limit = (basis.ulx(l, 1)+basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            else:
                unl_limit = (basis.d_ulx(l, 1, 1)-basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            nvec = self._unl_even_ref[:,0]
            if statistics == "f":
                unl_coeff = nvec*unl.imag
            else:
                unl_coeff = nvec*nvec*unl.real
            dunl_coeff = abs(unl_limit-unl_coeff[len(unl_coeff)-1]) \
                    if abs(unl_limit) < 1e-12 \
                    else abs(unl_limit-unl_coeff[len(unl_coeff)-1])/abs(unl_limit)
            dunl_limit = numpy.append(dunl_limit, dunl_coeff)
            return (abs(dunl.max()), abs(dunl_limit.max()))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_unl(self):
        for _lambda in [10.0, 10000.0]:
            for _statistics in ["f", "b"]:
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                rf_ref = refdata("../unl_safe_ref.h5", prefix)
                basis = ir.basis("../irbasis.h5", prefix+"_np8")
                diff = rf_ref.check_data(basis, _statistics)
                self.assertLessEqual(diff[0], 1e-8)
                self.assertLessEqual(diff[1], 1e-7)


if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
