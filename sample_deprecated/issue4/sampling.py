from __future__ import print_function

import numpy
import irlib
def find_zeros(ulx):
    Nx = 10000
    eps = 1e-10
    tvec = numpy.linspace(-3, 3, Nx) #3 is a very safe option.
    xvec = numpy.tanh(0.5*numpy.pi*numpy.sinh(tvec))

    zeros = []
    for i in range(Nx-1):
        if ulx(xvec[i]) * ulx(xvec[i+1]) < 0:
            a = xvec[i+1]
            b = xvec[i]
            u_a = ulx(a)
            u_b = ulx(b)
            while a-b > eps:
                #print(a,b)
                half_point = 0.5*(a+b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5*(a+b))
    return numpy.array(zeros)


for Lambda in [1000.0]:
    basis = irlib.loadtxt("/Users/hiroshi/ClionProjects/ir_basis/samples/np10/basis_f-mp-Lambda"+str(Lambda)+".txt")
    Nl = basis.dim()
    print("s_l/s_0 ", basis.sl(Nl-1)/basis.sl(0))

    sl = numpy.array([basis.sl(l)/basis.sl(0) for l in range(Nl)])

    zeros_ulx = find_zeros(lambda x : basis.ulx(Nl-1,x)) 

    sampling_points = numpy.linspace(-1, zeros_ulx[0], 10)

    for x in sampling_points:
        # basis.ulx(x,l) return a float but it uses multi-precision math internally for interpolation.
        str_x = "{:.20f}".format(x)
        print(str_x, "{:.20f}".format(basis.ulx(Nl-1,x)))
        print(str_x, basis.ulx_str(Nl-1,str_x))

