import h5py
import numpy

with h5py.File("hdf5_test.h5", "w") as f:
    f["/test_data/double"] = 100.0
    #f["/test_data/c_str"] = "FG"
    f["/test_data/int"] = 100
    N1 = 2
    N2 = 3
    N3 = 4
    f["/test_data/double_array1"] = numpy.arange(N1, dtype=float)
    f["/test_data/double_array2"] = numpy.arange(N1 * N2, dtype=float).reshape((N1,N2))
    f["/test_data/double_array3"] = numpy.arange(N1 * N2 * N3, dtype=float).reshape((N1,N2,N3))
