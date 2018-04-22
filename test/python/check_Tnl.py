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

            self._Tnl_ref = f[prefix+'/data/Tnl'].value
            self._Tnl_ref_max = f[prefix+'/data/Tnl_max'].value

    def check_data(self, basis, prefix):
            ndim = basis.dim() if basis.dim()%2 ==0 else basis.dim()-1
            Tnl = basis.compute_Tnl(self._Tnl_ref[:, 0])[:, ndim-1]
            dTnl = abs(Tnl- self._Tnl_ref[:,1])/self._Tnl_ref_max
            
            nTnl = [ _data[0]*_data[1].imag for _data in self._Tnl_ref]
            Tnl_limit_imag = (basis.ulx(ndim-1, 1)+basis.ulx(ndim-1, -1))/(numpy.pi*math.sqrt(2.0))
            dnTnl_imag = abs(Tnl_limit_imag-nTnl[len(nTnl)-1]) \
                    if abs(Tnl_limit_imag) < math.pow(10.0, -12.0) \
                    else abs(Tnl_limit_imag-nTnl[len(nTnl)-1])/Tnl_limit_imag
            dTnl = numpy.append(dTnl, dnTnl_imag)

            nnTnl = [ -_data[0]*_data[0]*_data[1].real for _data in self._Tnl_ref]
            Tnl_limit_real = (basis.d_ulx(ndim-1, 1, 1)+basis.d_ulx(ndim-1, -1, 1))/(numpy.pi*numpy.pi*math.sqrt(2.0))
            dnnTnl_real = abs(Tnl_limit_real-nnTnl[len(nnTnl)-1]) \
                    if abs(Tnl_limit_real) < math.pow(10.0, -12.0) \
                    else abs(Tnl_limit_real-nnTnl[len(nnTnl)-1])/Tnl_limit_real
            dTnl = numpy.append(dTnl, dnnTnl_real)            
            return numpy.max(dTnl)

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Tnl(self):
        for _lambda in [10.0, 1000.0]:
            for _statistics in ["f", "b"]:
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                rf_ref = refdata("../tnl_safe_ref.h5", prefix)            
                basis = ir.basis("../irbasis.h5", prefix)
                self.assertLessEqual(rf_ref.check_data(basis, prefix), math.pow(10.0, -8))

if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
