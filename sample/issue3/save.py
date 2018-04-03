import os
import argparse
import numpy as np
import h5py
import irlib

class BasisSet(object):
    def __init__(self, _h5file, prefix_name):
        self._h5file = h5file
        self._prefix_name = prefix_name

        # Comment: No need to create group explicitly
        #if self._prefix_name in self._h5file:
            #del self._h5file[self._prefix_name]

        #self._h5file.create_group(self._prefix_name)
        #self._h5file.create_group(self._prefix_name+"/info")
        #self._h5file.create_group(self._prefix_name+"/ulx")
        #self._h5file.create_group(self._prefix_name+"/vly")
        #self._h5file._create_group(self._prefix_name)

    def _write_data(self, path, data):
        if path in self._h5file:
            del self._h5file[path]

        self._h5file[path] = data

    def set_info(self, Lambda, dim, statistics):
        self._write_data(self._prefix_name+"/info/Lambda", Lambda)
        self._write_data(self._prefix_name+"/info/dim", dim)
        self._write_data(self._prefix_name+"/info/statistics", statistics)

    def set_sl(self, sl):
        #CheckSizet
        dir = self._prefix_name
        self._write_data(dir+"/sl", sl)
        return True

    def set_func(self, func_name, data, np, ns, section_edges):

        if func_name != "ulx" and func_name != "vly":
            print("Error in Set_func: func_name must be ulx or vly.")
            return False

        #TODO: CheckSize
        dir = self._prefix_name+"/"+func_name
        self._write_data(dir+"/ns", data=ns)
        self._write_data(dir+"/np", data=np)

        assert data.shape[1] == section_edges.size-1
        assert data.shape[2] == np

        self._write_data(dir+"/data", data = data)
        self._write_data(dir+"/section_edges", data = section_edges)

        return True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='save.py',
        description='Output results to hdf5 file.',
        epilog='end',
        add_help=True,
    )

    parser.add_argument('-o', '--output', action='store', dest='outputfile',
                        default='irbasis.h5',
                        type=str, choices = None,
                        help=('Path to output hdf5 file.'),
                        metavar=None)
    parser.add_argument('-i', '--input', action='store', dest='inputfile',
                        type=str, choices = None,
                        required = True,
                        help=('Path to input file.'),
                        metavar=None)
    parser.add_argument('-l', '--lambda', action='store', dest='lambda',
                        required = True,
                        type=float, choices = None,
                        help=('Value of lambda.'),
                        metavar=None)
    parser.add_argument('-p', '--prefix', action='store', dest='prefix',
                        type=str, choices = None,
                        default = '/',
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
    irset.set_info(b.Lambda(), nl, b.get_statistics())

    sl = np.array([b.sl(i) for i in range(0, nl)])
    irset.set_sl(sl)

    # input ulx
    ns = b.num_sections_ulx()
    n_local_poly = b.num_local_poly_ulx()
    coeff = np.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_ulx(l, s, p)
    section_edge_ulx = np.array([b.section_edge_ulx(i) for i in range(ns+1)])
    irset.set_func("ulx", coeff, n_local_poly, ns, section_edge_ulx )

    # input vly
    ns = b.num_sections_vly()
    n_local_poly = b.num_local_poly_vly()
    coeff = np.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_vly(l, s, p)
    section_edge_vly = np.array([b.section_edge_vly(i) for i in range(ns+1)])
    irset.set_func("vly", coeff, n_local_poly, ns, section_edge_vly )
    
    h5file.flush()
    h5file.close()
