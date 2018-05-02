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

            self._Tnl_odd_ref = f[prefix+'/data/lodd/Tnl'].value
            self._Tnl_odd_ref_max = f[prefix+'/data/lodd/Tnlmax'].value
            self._Tnl_odd_l = f[prefix+'/data/lodd/l'].value
            
            self._Tnl_even_ref = f[prefix+'/data/leven/Tnl'].value
            self._Tnl_even_ref_max = f[prefix+'/data/leven/Tnlmax'].value
            self._Tnl_even_l = f[prefix+'/data/leven/l'].value
            
            
    def check_data(self, basis, prefix, statistics):
            #Check odd-l
            l = self._Tnl_odd_l 
            Tnl = basis.compute_Tnl(self._Tnl_odd_ref[:, 0])[:, l]
            dTnl = abs(Tnl- self._Tnl_odd_ref[:,1])/self._Tnl_odd_ref_max

            if statistics == "f":
                Tnl_limit = -(basis.d_ulx(l, 1, 1)+basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            else:
                Tnl_limit = -(basis.ulx(l, 1)-basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            nvec = self._Tnl_odd_ref[:,0]
            if statistics == "f":
                Tnl_coeff = nvec*nvec*Tnl.real
            else:
                Tnl_coeff = nvec*Tnl.imag
            dTnl_coeff= abs(Tnl_limit-Tnl_coeff[len(Tnl_coeff)-1]) \
                    if abs(Tnl_limit) < math.pow(10.0, -12) \
                    else abs(Tnl_limit-Tnl_coeff[len(Tnl_coeff)-1])/abs(Tnl_limit)
            dTnl_limit =dTnl_coeff

            #Check even-l
            l = self._Tnl_even_l 
            Tnl = basis.compute_Tnl(self._Tnl_even_ref[:, 0])[:, l]
            dTnl = numpy.append(dTnl, abs(Tnl- self._Tnl_even_ref[:,1])/self._Tnl_even_ref_max)
            
            if statistics == "f":
                Tnl_limit = (basis.ulx(l, 1)+basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            else:
                Tnl_limit = (basis.d_ulx(l, 1, 1)-basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            nvec = self._Tnl_even_ref[:,0]
            if statistics == "f":
                Tnl_coeff = nvec*Tnl.imag
            else:
                Tnl_coeff = nvec*nvec*Tnl.real
            dTnl_coeff = abs(Tnl_limit-Tnl_coeff[len(Tnl_coeff)-1]) \
                    if abs(Tnl_limit) < 1e-12 \
                    else abs(Tnl_limit-Tnl_coeff[len(Tnl_coeff)-1])/abs(Tnl_limit)
            dTnl_limit = numpy.append(dTnl_limit, dTnl_coeff)            
            return (abs(dTnl.max()), abs(dTnl_limit.max()))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Tnl(self):
        for _lambda in [10.0, 10000.0]:
            for _statistics in ["f", "b"]:
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                rf_ref = refdata("../tnl_safe_ref.h5", prefix)            
                basis = ir.basis("../irbasis.h5", prefix)
                diff = rf_ref.check_data(basis, prefix, _statistics)
                self.assertLessEqual(diff[0], math.pow(10.0, -8))
                self.assertLessEqual(diff[1], math.pow(10.0, -7))

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
