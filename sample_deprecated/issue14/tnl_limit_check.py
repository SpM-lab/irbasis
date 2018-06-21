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

    def set_tnl(self, tnl):
        dir = self._prefix_name
        self._write_data(dir + "/data/Tnl", tnl)
        max_value = numpy.max(abs(tnl[:,1]))
        self._write_data(dir + "/data/Tnl_max", max_value)

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

    nvec_long = numpy.array([10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10])
    nvec = nvec_long
    Nl = b.dim() if b.dim()%2==0 else b.dim()-1
    print([b.ulx_derivative(Nl-1, 1, k) for k in range(4)])
    Tnl = numpy.array([ (b.compute_Tnl_safe(int(n), Nl-1)*(n+0.5)*(n+0.5)) for n in nvec])
    print(Tnl)
    for i in range(len(nvec_long)):
        print(nvec_long[i], Tnl[i].real)
    
    print(-(b.ulx_derivative(Nl-1, 1, 1)+b.ulx_derivative(Nl-1, -1, 1))/(numpy.pi*numpy.pi*sqrt(2.0)))

    
