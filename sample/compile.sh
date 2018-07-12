# Here, we like the program to the HDF5 C library installed in /usr/local/lib.
# We define the NDEBUG macro to disable assertions.
# If hdf5.h is not found at compile time, please tell the compiler where that header file is by using "-I" option.
g++ api.cpp -I /usr/local/include -L /usr/local/lib -lhdf5 -DNDEBUG -O3
