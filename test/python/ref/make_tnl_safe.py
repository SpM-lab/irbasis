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

    def set_tnl(self, tnl, l):
        dir = self._prefix_name
        str_dir = "leven" if l%2 == 0 else "lodd"
        self._write_data(dir + "/data/"+str_dir+"/Tnl", tnl)
        self._write_data(dir + "/data/"+str_dir+"/l", l)
        self._write_data(dir + "/data/"+str_dir+"/Tnlmax", numpy.amax(abs(tnl[:,1])))

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
    nvec_short = numpy.arange(100)
    nvec_long = numpy.array([10**3, 10**4, 10**5, 10**6, 10**7, 10**8])
    nvec = numpy.append(nvec_short, nvec_long)
    Nodd_max = b.dim()-1 if b.dim()%2==0 else b.dim()-2
    Tnl_odd = numpy.array([ (int(n), b.compute_Tnl_safe(int(n), Nodd_max)) for n in nvec])
    irset.set_tnl(Tnl_odd, Nodd_max)
    Neven_max = b.dim()-1 if b.dim()%2==1 else b.dim()-2
    Tnl_even = numpy.array([ (int(n), b.compute_Tnl_safe(int(n), Neven_max)) for n in nvec])
    irset.set_tnl(Tnl_even, Neven_max)
    
