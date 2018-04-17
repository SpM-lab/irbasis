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
        // see https://www.physics.ohio-state.edu/~wilkins/computing/HDF/hdf5tutorial/examples/C/h5_rdwt.c
        
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

        // To be implemented ...
        //inline void hdf5_read_double_array1(hid_t& file, const std::string& name, std::size_t& size, std::vector<double>& data);
        //inline void hdf5_read_double_array2(hid_t& file, const std::string& name, std::vector<std::size_t>& size, std::vector<double>& data);
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
