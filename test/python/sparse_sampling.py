from __future__ import print_function
from builtins import range

import unittest
import numpy
import scipy
import irbasis

Lambda = 1E+7

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_sampling_point_matsubara(self):
        for stat in ['F', 'B']:
            b = irbasis.load(stat, Lambda, "../irbasis.h5")

            dim = b.dim()
            whichl = dim - 1
            sp = irbasis.sampling_points_matsubara(b, whichl)
            if stat == 'F':
                assert numpy.all([-s-1 in sp for s in sp])
            elif stat in ['B']:
                assert numpy.all([-s in sp for s in sp])

            assert len(sp) >= whichl + 1

            Unl = b.compute_unl(sp)[:, :dim]
            U, S, Vh = scipy.linalg.svd(Unl, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

    def test_sampling_point_x(self):
        for stat in ['F', 'B']:
            b = irbasis.load(stat, Lambda, "../irbasis.h5")

            dim = b.dim()
            whichl = dim - 1
            sp = irbasis.sampling_points_x(b, whichl)
            assert len(sp) == whichl+1
            uxl = numpy.array([b.ulx(l, x) for l in range(dim) for x in sp]).reshape((dim, dim))
            U, S, Vh = scipy.linalg.svd(uxl, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

    def test_sampling_point_y(self):
        for stat in ['F', 'B']:
            b = irbasis.load(stat, Lambda, "../irbasis.h5")

            dim = b.dim()
            whichl = dim - 1
            sp = irbasis.sampling_points_y(b, whichl)
            #print(len(sp), whichl)
            #print(sp)
            assert len(sp) == whichl+1

if __name__ == '__main__':
    unittest.main()
