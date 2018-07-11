from __future__ import print_function
import numpy
import irbasis

# By default, Lambda = 10, 100, 1000, 10000 are available.
Lambda = 1000.0

# Fermionic
# If you has not installed the irbasis package via pip,
# you must specify the location of a data file as follows.
#   irbasis.load('F',  Lambda, './irbasis.h5')
basis = irbasis.load('F', Lambda)

l = 0
print("l =",l,",Lambda =", Lambda)

x = 1
y = 1

# Dimensions of basis
print("Dim ", basis.dim())

# u_0(x = 1) and v_0(y = 1) 
print("ulx ", basis.ulx(l,x))
print("vly ", basis.vly(l,y))

# Singular value s_0
print("sl ", basis.sl(l))

# The k-th derivative of u_l(x) and v_l(y)  (k = 1,2,3)
for k in [1, 2, 3]:
    print("k = ", k)
    print("d_ulx ", basis.d_ulx(l,x,k))
    print("d_vly ", basis.d_vly(l,y,k))

# Fourier transform of ulx
n = numpy.arange(1000)
unl = basis.compute_unl(n)
print("dimensions of unl ", unl.shape)
