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

        // Simple implementation without meta programming...
        template<typename T, int DIM>
        class multi_array {
            public:
                multi_array(int N1, int N2) : data_(N1*N2) {
                    assert(DIM == 2);
                    extents_[0] = N1;
                    extents_[1] = N2;
                }

                multi_array(int N1, int N2, int N3) : data_(N1*N2*N3) {
                    assert(DIM == 3);
                    extents_[0] = N1;
                    extents_[1] = N2;
                    extents_[2] = N3;
                }

                std::size_t extent(int i) const {
                    assert(i >= 0);
                    assert(i < DIM);
                    return extents_[i];
                }

                T& operator()(int i, int j) {
                    assert (DIM == 2);
                    int idx = extents_[1] * i + j;
                    assert(idx >= 0 && idx < data_.size());
                    return data_[idx];
                }

                const T& operator()(int i, int j) const {
                    assert (DIM == 2);
                    int idx = extents_[1] * i + j;
                    assert(idx >= 0 && idx < data_.size());
                    return data_[idx];
                }

                T& operator()(int i, int j, int k) {
                    assert (DIM == 3);
                    int idx = extents_[1] * extents_[2] * i + extents_[2] * j + k;
                    assert(idx >= 0 && idx < data_.size());
                    return data_[idx];
                }

                const T& operator()(int i, int j, int k) const {
                    assert (DIM == 3);
                    int idx = extents_[1] * extents_[2] * i + extents_[2] * j + k;
                    assert(idx >= 0 && idx < data_.size());
                    return data_[idx];
                }

            private:
                std::vector<T> data_;
                std::size_t extents_[DIM];
        };
        
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
