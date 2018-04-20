from __future__ import print_function
from builtins import range

import unittest
import numpy
import h5py
import platform
import irbasis as ir

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
            self._tnl_ref_max = f[prefix+'/data/Tnl_max'].value

    def check_data(self, basis, prefix):
            rf = ir.basis("../irbasis.h5", prefix)
            ndim = rf.dim() if rf.dim()%2 ==0 else rf.dim()-1
            Tnl = rf.compute_Tnl(self._Tnl_ref[:, 0])[:, ndim-1]
            print(Tnl, self._Tnl_ref[:,1])
            #for _tnl_ref in self._Tnl_ref:
             #   n=int(_tnl_ref[0])
              #  Tnl = rf.compute_Tnl(n)
               # print(Tnl.size)
                
           
            exit(0)
            
class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Tnl(self):
        for _lambda in [10.0, 1000.0]:
            for _statistics in ["f", "b"]:
                prefix = "basis_"+_statistics+"-mp-Lambda"+str(_lambda)
                rf_ref = refdata("../tnl_safe_ref.h5", prefix)            
                basis = ir.basis("../irbasis.h5", prefix)
                rf_ref.check_data(basis, prefix)



if __name__ == '__main__':
    unittest.main()

# https://cmake.org/pipermail/cmake/2012-May/050120.html
