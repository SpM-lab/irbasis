import numpy
from mpmath import *

mp.dps = 100

a = mp.mpf("0.9999999")
b = mp.mpf("0.999999")

a_dd =  numpy.array((1,   -0.0000001))
b_dd =  numpy.array((1,   -0.000001))

diff = a-b
diff_float = float(a)-float(b)

print(a_dd - b_dd)

print(diff, diff_float)
