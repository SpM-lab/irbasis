from __future__ import print_function

import irbasis

b = irbasis.load("F", 1000.0, "./irbasis.h5")

print(b.dim())
