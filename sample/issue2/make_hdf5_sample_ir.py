import argparse
import numpy as np
from numpy.random import *
import h5py

parser = argparse.ArgumentParser(
    prog='make_hdf5_sample_ir.py',
    description='make sample hdf5 file for irlib.',
    epilog='end',
    add_help=True,
)

parser.add_argument('-o', '--output', action='store', dest='outputfile',
                    default='sample.h5',
                    type=str, choices=None,
                    help=('Path to hdf5 file.'),
                    metavar=None)

parser.add_argument('np', type=int)
parser.add_argument('Lambda', type=float)
parser.add_argument('stastics', type=str)

args = parser.parse_args()
h5file = h5py.File(args.outputfile, "a")
np = args.np
Lambda = args.Lambda
stastics = args.stastics


dir = stastics
subdir="/np_"+str(np)+"_Lambda_"+str(Lambda)
h5file.create_group(dir+subdir)
Data_type=["/ulx", "/vly", "/sl"]

N=10
np=5


for _type in Data_type:
    subsubdir=dir+subdir+_type
    if _type != "/sl":
        h5file.create_group(dir+subdir+_type)
        h5file.create_dataset(subsubdir+"/N", data=N)
        for i in range(0, N):
            h5file.create_dataset(subsubdir+"/"+str(i), data=rand(np))    
    else:
        h5file.create_dataset(dir+subdir+_type, data=rand(np))

        
h5file.flush()
h5file.close()
