#import argparse
import numpy 
from numpy.random import *
import h5py
import sys

#parser = argparse.ArgumentParser(
    #prog='make_hdf5_sample_ir.py',
    ##description='make sample hdf5 file for irlib.',
    #epilog='end',
    #add_help=True,
#)
#
#parser.add_argument('-o', '--output', action='store', dest='outputfile',
                    #default='sample.h5',
                    #type=str, choices=None,
                    #help=('Path to hdf5 file.'),
                    #metavar=None)
#
#parser.add_argument('np', type=int)
#parser.add_argument('Lambda', type=float)
#parser.add_argument('stastics', type=str)
#
#args = parser.parse_args()
h5file = h5py.File("template.h5", "w")

#np = args.np
#Lambda = args.Lambda
#stastics = args.stastics

name = "name"

dim = 30    # Number of basis functions
np_x = 10     # Number of polynomials for each section
ns_x = 100    # Number of sections    
np_y = 10     # Number of polynomials for each section
ns_y = 100    # Number of sections    
Lambda = 100.0
stastics = 'FERMION'
#augmented = False

#dir = stastics
topdir="/"+'prefix'
h5file.create_group(topdir)
Data_type=["/ulx", "/vly", "/sl"]

# Info
h5file[topdir+"/info/Lambda"] = Lambda
h5file[topdir+"/info/stastics"] = stastics
#h5file[topdir+"/info/augmented"] = augmented
h5file[topdir+"/info/dim"] = dim

# sl
h5file[topdir+"/sl/data"] = numpy.zeros((dim), dtype=float)

# ulx
h5file[topdir+"/ulx/np"] = np_x
h5file[topdir+"/ulx/ns"] = ns_x
h5file[topdir+"/ulx/data"] = numpy.zeros((dim,ns_x,np_x), dtype=float)

# vly
h5file[topdir+"/vly/np"] = np_y
h5file[topdir+"/vly/ns"] = ns_y
h5file[topdir+"/vly/data"] = numpy.zeros((dim,ns_y,np_y), dtype=float)

sys.exit(1)

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
