import h5py

with h5py.File("hdf5_test.h5", "w") as f:
    f["/test_data/double"] = 100.0
    #f["/test_data/char"] = "F"
    f["/test_data/int"] = 100
