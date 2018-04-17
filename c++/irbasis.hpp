#pragma once

#include <iostream>
#include <complex>
//#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>
#include <fstream>
#include <numeric>

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
        
        // read array of double
        template<int DIM>
        void hdf5_read_double_array(hid_t& file, const std::string& name, std::vector<std::size_t>& extents, std::vector<double>& data) {
            hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
            hid_t space = H5Dget_space(dataset);
            std::vector<hsize_t> dims(DIM);
            int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
            assert(n_dims == DIM);
            std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<double>());
            data.resize(tot_size);
            extents.resize(DIM);
            for (int i=0; i<DIM; ++i) {
                extents[i] = static_cast<std::size_t>(dims[i]);
            }
            H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
            H5Dclose(dataset);
        }
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
