import os
import argparse
import numpy as np
import h5py
import irlib

class basis_set(object):
    def __init__(self, _h5file, _prefix_name):
        self.h5file=_h5file
        self.prefix_name=_prefix_name
        self.h5file.create_group(self.prefix_name)
        self.h5file.create_group(self.prefix_name+"/info")
        self.h5file.create_group(self.prefix_name+"/ulx")
        self.h5file.create_group(self.prefix_name+"/vly")
        
    def Set_sl(self, _sl):
        #CheckSize
        dir = self.prefix_name+"/info"
        self.h5file.create_dataset(dir+"/sl", data=_sl)
        return True

    def Set_func(self, _func_name, _data, _np, _ns, _section_edges):

        if _func_name != "ulx" and _func_name != "vly":
            print("Error in Set_func: _func_name must be ulx or vly.")
            return False

        #TODO: CheckSize
        dir = self.prefix_name+"/"+_func_name
        self.h5file.create_dataset(dir+"/ns", data=_ns)
        self.h5file.create_dataset(dir+"/np", data=_np)
        if _data.shape[1] != _np : return false
        if _data.shape[2] != _section_edges.size : return false
        self.h5file.create_dataset(dir+"/data", data=_data)
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

    args = parser.parse_args()
    if os.path.exists(args.inputfile):
        b = irlib.loadtxt(args.inputfile)
    else:
        print("Input file does not exist.")
        exit(-1)
    h5file = h5py.File(args.outputfile, "a")
    irset=basis_set(h5file, args.inputfile)
    nl = b.dim()
    sl = np.array([b.sl(i) for i in range(0, nl)])
    irset.Set_sl(sl)

    #input ulx
    ns = b.num_sections_ulx()
    n_local_poly = b.num_local_poly_ulx()
    coeff = np.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_ulx(l, s, p)
    section_edge_ulx = np.array([b.section_edge_ulx(i) for i in range(ns)])
    irset.Set_func("ulx", coeff, n_local_poly, ns, section_edge_ulx )

    #input vly
    ns = b.num_sections_vly()
    n_local_poly = b.num_local_poly_vly()
    coeff = np.zeros((nl, ns, n_local_poly), dtype=float)
    for l in range(nl):
        for s in range(ns):
            for p in range(n_local_poly):
                coeff[l, s, p] = b.coeff_vly(l, s, p)
    section_edge_vly = np.array([b.section_edge_vly(i) for i in range(ns)])
    irset.Set_func("vly", coeff, n_local_poly, ns, section_edge_vly )
    
    h5file.flush()
    h5file.close()
