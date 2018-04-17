#pragma once

#include <iostream>
#include <complex>
//#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>
#include <fstream>

#include <hdf5.h>

namespace {

    namespace internal {
        // https://www.physics.ohio-state.edu/~wilkins/computing/HDF/hdf5tutorial/examples/C/h5_rdwt.c
        // https://support.hdfgroup.org/ftp/HDF5/current/src/unpacked/examples/h5_read.c
        
        // read a double
        inline double hdf5_read_double(hid_t& file, const std::string& name) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            double data;
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
            H5Dclose(dataset);
            return data;
        }

        // read an int
        inline int hdf5_read_int(hid_t& file, const std::string& name) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            int data;
            H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
            H5Dclose(dataset);
            return data;
        }
        
        // read 1D array of double
        inline void hdf5_read_double_array1(hid_t& file, const std::string& name, std::vector<double>& data) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            hid_t space = H5Dget_space(dataset);
            hsize_t dims[100];
            int n_dims = H5Sget_simple_extent_dims(space, dims, NULL);
            assert(n_dims == 1);
            data.resize(dims[0]);
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
            H5Dclose(dataset);
        }

        // read 2D array of double
        inline void hdf5_read_double_array2(hid_t& file, const std::string& name, std::vector<std::size_t>& extents, std::vector<double>& data) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            hid_t space = H5Dget_space(dataset);
            hsize_t dims[100];
            int n_dims = H5Sget_simple_extent_dims(space, dims, NULL);
            assert(n_dims == 2);
            data.resize(dims[0] * dims[1]);
            extents.resize(2);
            for (int i=0; i<2; ++i) {
                extents[i] = static_cast<std::size_t>(dims[i]);
            }
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
            H5Dclose(dataset);
        }

        //
        // read 3D array of double
        inline void hdf5_read_double_array3(hid_t& file, const std::string& name, std::vector<std::size_t>& extents, std::vector<double>& data) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            hid_t space = H5Dget_space(dataset);
            hsize_t dims[100];
            int n_dims = H5Sget_simple_extent_dims(space, dims, NULL);
            assert(n_dims == 3);
            data.resize(dims[0] * dims[1] * dims[2]);
            extents.resize(3);
            for (int i=0; i<3; ++i) {
                extents[i] = static_cast<std::size_t>(dims[i]);
            }
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
            H5Dclose(dataset);
        }

        // To be implemented ...
        //inline void hdf5_read_double_array3(hid_t& file, const std::string& name, std::vector<std::size_t>& size, std::vector<double>& data);
    }

    /*
    class basis {
    public:
        basis(
            const std::string& file_name,
            const std::string& prefix = "",
        ) throw(std::runtime_error) {
        }

    };
    */

}
