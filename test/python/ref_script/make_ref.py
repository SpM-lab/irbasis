from __future__ import print_function

import numpy 
import irlib
import scipy.integrate as integrate

from mpmath import *

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

def GetMaxAbsValue(basis, type):
    Nl = basis.dim()
    # Consider even number
    if Nl % 2 == 1: Nl -= 1
    if type == "ulx":
        func_l = (lambda x : basis.ulx(Nl-1,x))
        func_l_derivative = (lambda x : basis.ulx_derivative(Nl-1,x,1))
    elif type == "vly":
        func_l = (lambda x : basis.vly(Nl-1,x))
        func_l_derivative = (lambda x : basis.vly_derivative(Nl-1,x,1))
    else:
        return None
    zeros_data=find_zeros(func_l_derivative)
    values_zeros = numpy.array( [ abs(func_l(_x)) for _x in zeros_data] )
    max_index = numpy.argmax(values_zeros)
    max_point=zeros_data[max_index]
    if abs(func_l(1.0)) > values_zeros[max_index]:
        max_point = 1.0
    elif abs(func_l(-1.0)) > values_zeros[max_index]:
        max_point = -1.0
    return (max_point, abs(func_l(max_point)))

def SaveValues(Lambda, statistics):
    basis = irlib.loadtxt("np10/basis_"+statistics+"-mp-Lambda"+str(Lambda)+".txt")
    Nl = basis.dim()
    if Nl % 2 == 1 : Nl-=1    
    points=GetMaxAbsValue(basis, "ulx")
    if points is not None: print("(\"ulx\", ", Lambda, ") : ", points)
    edges = numpy.array([basis.section_edge_ulx(s) for s in range(basis.num_sections_ulx()+1)])
    Region0 = numpy.linspace(edges[0], edges[1], 10)
    values0 = numpy.array( [ basis.ulx(Nl-1, _x) for _x in Region0] )
    Region1 = numpy.linspace(edges[basis.num_sections_ulx()-1], edges[basis.num_sections_ulx()], 10)
    values1 = numpy.array( [ basis.ulx(Nl-1, _x) for _x in Region1] )    
    numpy.savetxt("output_"+statistics+"-mp-Lambda"+str(Lambda)+"_ulx.csv", [Region0, values0, Region1, values1], delimiter=',')

    points=GetMaxAbsValue(basis, "vly")
    if points is not None: print("(\"vly\", ", Lambda, ") : ", points)
    edges = numpy.array([basis.section_edge_vly(s) for s in range(basis.num_sections_vly()+1)])
    Region0 = numpy.linspace(edges[0], edges[1], 10)
    values0 = numpy.array( [ basis.vly(Nl-1, _y) for _y in Region0] )
    Region1 = numpy.linspace(edges[basis.num_sections_vly()-1], edges[basis.num_sections_vly()], 10)
    values1 = numpy.array( [ basis.vly(Nl-1, _y) for _y in Region1] )    
    numpy.savetxt("output_"+statistics+"-mp-Lambda"+str(Lambda)+"_vly.csv", [Region0, values0, Region1, values1], delimiter=',')    

N = 1000
xvec = numpy.linspace(-1, 1, N)

## Construct basis
idx = 0
for Lambda in [10.0, 10000.0]:
    SaveValues(Lambda, statistics="f")
    SaveValues(Lambda, statistics="b")
