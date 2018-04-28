from __future__ import print_function
from builtins import range

import unittest
import numpy
import h5py
import platform
import irbasis as ir
import math

is_python3 = int(platform.python_version_tuple()[0]) == 3

def _from_bytes_to_utf8(s):
    """
    from bytes to string
    :param s:
    :return:
    """
    if is_python3 and isinstance(s, bytes):
        return s.decode('utf-8')
    else:
        return s


class refdata(object):
    def __init__(self, file_name, prefix=""):
        
        with h5py.File(file_name, 'r') as f:
            self._Lambda = f[prefix+'/info/Lambda'].value
            self._dim = f[prefix+'/info/dim'].value
            self._statistics = _from_bytes_to_utf8(f[prefix+'/info/statistics'].value)  # from bytes to string

            self._Tnl_odd_ref = f[prefix+'/data/lodd/Tnl'].value
            self._Tnl_odd_ref_max = f[prefix+'/data/lodd/Tnlmax'].value
            self._Tnl_odd_l = f[prefix+'/data/lodd/l'].value
            
            self._Tnl_even_ref = f[prefix+'/data/leven/Tnl'].value
            self._Tnl_even_ref_max = f[prefix+'/data/leven/Tnlmax'].value
            self._Tnl_even_l = f[prefix+'/data/leven/l'].value
            
            
    def check_data(self, basis, prefix, statistics):
            ndim = basis.dim() if basis.dim()%2 ==0 else basis.dim()-1
            #Check odd-l
            l = self._Tnl_odd_l 
            Tnl = basis.compute_Tnl(self._Tnl_odd_ref[:, 0])[:, l]
            dTnl = abs(Tnl- self._Tnl_odd_ref[:,1])/self._Tnl_odd_ref_max

            if statistics == "f":
                Tnl_limit_real = (basis.d_ulx(l, 1, 1)+basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            else:
                Tnl_limit_real = (basis.d_ulx(l, 1, 1)-basis.d_ulx(l, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            #nnTnl = [ -_data[0]*_data[0]*_data[1].real for _data in self._Tnl_odd_ref]
            nvec = self._Tnl_odd_ref[:,0]
            nnTnl = -nvec*nvec*Tnl
            print(nnTnl, Tnl_limit_real)
            dnnTnl_real= abs(Tnl_limit_real-nnTnl[len(nnTnl)-1]) \
                    if abs(Tnl_limit_real) < 1e-12 \
                    else abs(Tnl_limit_real-nnTnl[len(nnTnl)-1])/abs(Tnl_limit_real)
            dTnl =  numpy.append(dTnl, dnnTnl_real)

            #Check even-l
            l = self._Tnl_even_l 
            Tnl = basis.compute_Tnl(self._Tnl_even_ref[:, 0])[:, l]
            dTnl = numpy.append(dTnl, abs(Tnl- self._Tnl_even_ref[:,1])/self._Tnl_even_ref_max)
            if statistics == "f":
                Tnl_limit_imag = (basis.ulx(l, 1)+basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            else:
                Tnl_limit_imag = -(basis.ulx(l, 1)-basis.ulx(l, -1))/(numpy.pi*math.sqrt(2.0))
            #nTnl = [ _data[0]*_data[1].imag for _data in self._Tnl_even_ref]
            nvec = self._Tnl_even_ref[:,0]
            nTnl = -nvec*Tnl
            dnTnl_imag = abs(Tnl_limit_imag-nTnl[len(nTnl)-1]) \
                    if abs(Tnl_limit_imag) < 1e-12 \
                    else abs(Tnl_limit_imag-nTnl[len(nTnl)-1])/abs(Tnl_limit_imag)
            dTnl = numpy.append(dTnl, dnTnl_imag)
            return print(abs(dTnl.max()))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Tnl(self):
        for _lambda in [10.0, 100.0]:
            #for _statistics in ["f", "b"]:
            for _statistics in ["f", "b"]:
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                rf_ref = refdata("../tnl_safe_ref.h5", prefix)            
                basis = ir.basis("../irbasis.h5", prefix)
                print(1e-8)
                self.assertLessEqual(rf_ref.check_data(basis, prefix, _statistics), math.pow(10.0, -8))

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
