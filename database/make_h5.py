from __future__ import print_function

import os
import argparse
import numpy
import h5py
import irlib
import scipy.integrate as integrate

from mpmath import *


class BasisSet(object):
    def __init__(self, h5file, prefix_name):
        self._h5file = h5file
        self._prefix_name = prefix_name
        
    def _write_data(self, path, data):
        if path in self._h5file:
            del self._h5file[path]

        self._h5file[path] = data

    def set_info(self, Lambda, dim, statistics):
        self._write_data(self._prefix_name + "/info/Lambda", Lambda)
        self._write_data(self._prefix_name + "/info/dim", dim)
        self._write_data(self._prefix_name + "/info/statistics", \
                         0 if statistics == "B" else 1)
        self._dim = dim
        self._Lambda = Lambda
        
    def set_sl(self, sl):
        # CheckSizet
        dir = self._prefix_name
        self._write_data(dir + "/sl", sl)
        return True

    def set_func(self, func_name, data, np, ns, section_edges):

        if func_name != "ulx" and func_name != "vly":
            print("Error in Set_func: func_name must be ulx or vly.")
            return False

        # TODO: CheckSize
        dir = self._prefix_name + "/" + func_name
        self._write_data(dir + "/ns", data=ns)
        self._write_data(dir + "/np", data=np)

        assert data.shape[1] == section_edges.size - 1
        assert data.shape[2] == np

        self._write_data(dir + "/data", data=data)
        self._write_data(dir + "/section_edges", data=section_edges)

        return True

    def _find_zeros(self, ulx):
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

    def _get_max_abs_value(self, l, basis,  type):
        Nl = l
        if type == "ulx":
            func_l = (lambda x : basis.ulx(Nl-1,x))
            func_l_derivative = (lambda x : basis.ulx_derivative(Nl-1,x,1))
        elif type == "vly":
            func_l = (lambda x : basis.vly(Nl-1,x))
            func_l_derivative = (lambda x : basis.vly_derivative(Nl-1,x,1))
        else:
            return None
        zeros_data=self._find_zeros(func_l_derivative)
        values_zeros = numpy.array( [ abs(func_l(_x)) for _x in zeros_data] )
        max_index = numpy.argmax(values_zeros)
        max_point=zeros_data[max_index]
        if abs(func_l(1.0)) > values_zeros[max_index]:
            max_point = 1.0
        elif abs(func_l(-1.0)) > values_zeros[max_index]:
            max_point = -1.0
        return (int(l), max_point,  abs(func_l(max_point)))

    def save_ref_values(self, basis):
        Nl = self._dim
        Lambda = self._Lambda
        dir = self._prefix_name
 
        if Nl % 2 == 1 : Nl-=1    
        #Get ulx data
        points=self._get_max_abs_value(Nl, basis, "ulx")
        edges = numpy.array([basis.section_edge_ulx(s) for s in range(basis.num_sections_ulx()+1)])
        Region=numpy.append(numpy.linspace(edges[0], edges[1], 10),\
                            numpy.linspace(edges[basis.num_sections_ulx()-1], edges[basis.num_sections_ulx()], 10))
        ulx_data = numpy.array( [ (int(Nl), _x, basis.ulx(Nl-1, _x)) for _x in Region] )
        self._write_data(dir+"/ulx/ref/max", data=points)
        self._write_data(dir + "/ulx/ref/data", data=ulx_data)
        
        #Get vly data
        points=self._get_max_abs_value(Nl, basis, "vly")
        edges = numpy.array([basis.section_edge_vly(s) for s in range(basis.num_sections_vly()+1)])
        Region = numpy.append(numpy.linspace(edges[0], edges[1], 10),\
                              numpy.linspace(edges[basis.num_sections_vly()-1], edges[basis.num_sections_vly()], 10))
        vly_data = numpy.array( [ (int(Nl), _y, basis.vly(Nl-1, _y)) for _y in Region] )
        self._write_data(dir+"/vly/ref/max", data=points)
        self._write_data(dir+"/vly/ref/data", data=vly_data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='save.py',
        description='Output results to hdf5 file.',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-o', '--output', action='store', dest='outputfile',
                        default='irbasis.h5',
                        type=str, choices=None,
                        help=('Path to output hdf5 file.'),
                        metavar=None)
    parser.add_argument('-i', '--input', action='store', dest='inputfile',
                        type=str, choices=None,
                        required=True,
                        help=('Path to input file.'),
                        metavar=None)
    parser.add_argument('-l', '--lambda', action='store', dest='lambda',
                        required=True,
                        type=float, choices=None,
                        help=('Value of lambda.'),
                        metavar=None)
    parser.add_argument('-p', '--prefix', action='store', dest='prefix',
                        type=str, choices=None,
                        default='/',
                        help=('Data will be stored in this HF5 group.'),
                        metavar=None)

    args = parser.parse_args()
    if os.path.exists(args.inputfile):
        b = irlib.loadtxt(args.inputfile)
    else:
        print("Input file does not exist.")
        exit(-1)

    h5file = h5py.File(args.outputfile, "a")
    irset = BasisSet(h5file, args.prefix)
    nl = b.dim()

    # set info
    irset.set_info(b.Lambda(), nl, b.get_statistics_str())

    sl = numpy.array([b.sl(i) for i in range(0, nl)])
    irset.set_sl(sl)

    # input ulx
    ns = b.num_sections_ulx()
    n_local_poly = b.num_local_poly_ulx()
    coeff = numpy.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_ulx(l, s, p)
    section_edge_ulx = numpy.array([b.section_edge_ulx(i) for i in range(ns + 1)])
    irset.set_func("ulx", coeff, n_local_poly, ns, section_edge_ulx)

    # input vly
    ns = b.num_sections_vly()
    n_local_poly = b.num_local_poly_vly()
    coeff = numpy.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_vly(l, s, p)
    section_edge_vly = numpy.array([b.section_edge_vly(i) for i in range(ns + 1)])
    irset.set_func("vly", coeff, n_local_poly, ns, section_edge_vly)
    irset.save_ref_values(b)
    
    h5file.flush()
    h5file.close()
