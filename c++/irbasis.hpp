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
      multi_array() : data_(0) {
      }

      multi_array(int N1) : data_(N1) {
        assert(DIM == 1);
        extents_[0] = N1;
      }

      multi_array(int N1, int N2) : data_(N1 * N2) {
        assert(DIM == 2);
        extents_[0] = N1;
        extents_[1] = N2;
      }

      multi_array(int N1, int N2, int N3) : data_(N1 * N2 * N3) {
        assert(DIM == 3);
        extents_[0] = N1;
        extents_[1] = N2;
        extents_[2] = N3;
      }

      multi_array(std::size_t *dims) {
        resize(dims);
      }

      std::size_t extent(int i) const {
        assert(i >= 0);
        assert(i < DIM);
        return extents_[i];
      }

      void resize(std::size_t *dims) {
        std::size_t tot_size = std::accumulate(dims, dims + DIM, 1, std::multiplies<std::size_t>());
        data_.resize(tot_size);
        for (int i = 0; i < DIM; ++i) {
          extents_[i] = dims[i];
        }
      }

      std::size_t num_elements() const {
        return data_.size();
      }

      T *origin() {
        return &data_[0];
      }

      T &operator()(int i) {
        assert(DIM == 1);
        int idx = i;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

      const T &operator()(int i) const {
        assert(DIM == 1);
        int idx = i;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

      T &operator()(int i, int j) {
        assert(DIM == 2);
        int idx = extents_[1] * i + j;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

      const T &operator()(int i, int j) const {
        assert(DIM == 2);
        int idx = extents_[1] * i + j;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

      T &operator()(int i, int j, int k) {
        assert(DIM == 3);
        int idx = (i * extents_[1] + j) * extents_[2] + k;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

      const T &operator()(int i, int j, int k) const {
        assert(DIM == 3);
        int idx = (i * extents_[1] + j) * extents_[2] + k;
        assert(idx >= 0 && idx < data_.size());
        return data_[idx];
      }

    private:
      std::vector <T> data_;
      std::size_t extents_[DIM];
    };

    // https://www.physics.ohio-state.edu/~wilkins/computing/HDF/hdf5tutorial/examples/C/h5_rdwt.c
    // https://support.hdfgroup.org/ftp/HDF5/current/src/unpacked/examples/h5_read.c
    // read a double
    inline double hdf5_read_double(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      double data;
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
      H5Dclose(dataset);
      return data;
    }

    // read an int
    inline int hdf5_read_int(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      int data;
      H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
      H5Dclose(dataset);
      return data;
    }

    // read array of double
    template<int DIM>
    void hdf5_read_double_array(hid_t &file, const std::string &name, std::vector <std::size_t> &extents,
                                std::vector<double> &data) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      hid_t space = H5Dget_space(dataset);
      std::vector <hsize_t> dims(DIM);
      int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
      assert(n_dims == DIM);
      std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<double>());
      data.resize(tot_size);
      extents.resize(DIM);
      for (int i = 0; i < DIM; ++i) {
        extents[i] = static_cast<std::size_t>(dims[i]);
      }
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
      H5Dclose(dataset);
    }

    // read double multi_array
    template<int DIM>
    multi_array<double, DIM> load_multi_array(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      hid_t space = H5Dget_space(dataset);
      std::vector <hsize_t> dims(DIM);
      int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
      assert(n_dims == DIM);
      std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
      std::vector <std::size_t> extents(DIM);
      for (int i = 0; i < DIM; ++i) {
        extents[i] = static_cast<std::size_t>(dims[i]);
      }
      multi_array<double, DIM> a;
      a.resize(&extents[0]);
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.origin());
      std::vector<double> data(tot_size);
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
      multi_array<double, DIM> b = a;
      H5Dclose(dataset);

      return a;
    }

    // read int multi_array
    template<int DIM>
    multi_array<int, DIM> load_multi_iarray(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      hid_t space = H5Dget_space(dataset);
      std::vector <hsize_t> dims(DIM);
      int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
      assert(n_dims == DIM);
      std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
      std::vector <std::size_t> extents(DIM);
      for (int i = 0; i < DIM; ++i) {
        extents[i] = static_cast<std::size_t>(dims[i]);
      }
      multi_array<int, DIM> a;
      a.resize(&extents[0]);
      H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.origin());
      std::vector<int> data(tot_size);
      H5Dread(dataset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
      multi_array<int, DIM> b = a;
      H5Dclose(dataset);
      return a;
    }
  }

  struct func {
    internal::multi_array<int, 1> section_edges;
    internal::multi_array<double, 3> data;
    int np;
    int ns;
  };

  struct ref {
    internal::multi_array<double, 2> data;
    internal::multi_array<double, 1> max;
  };

  class basis {
  public:
    basis(
            const std::string &file_name,
            const std::string &prefix = ""
    ) throw(std::runtime_error) {
      hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
      //read info
      Lambda_ = internal::hdf5_read_double(file, prefix + std::string("/info/Lambda"));
      dim_ = internal::hdf5_read_int(file, prefix + std::string("/info/dim"));
      statistics_ = internal::hdf5_read_int(file, prefix + std::string("/info/statistics")) ? "B" : "F";

      //read sl
      sl_ = internal::load_multi_array<1>(file, prefix + std::string("/sl"));

      //read ulx
      ulx_.data = internal::load_multi_array<3>(file, prefix + std::string("/ulx/data"));
      ulx_.np = internal::hdf5_read_int(file, prefix + std::string("/ulx/np"));
      ulx_.ns = internal::hdf5_read_int(file, prefix + std::string("/ulx/ns"));
      ulx_.section_edges = internal::load_multi_iarray<1>(file, prefix + std::string("/ulx/section_edges"));

      //read ref_ulx
      ref_ulx_.data = internal::load_multi_array<2>(file, prefix + std::string("/ulx/ref/data"));
      ref_ulx_.max = internal::load_multi_array<1>(file, prefix + std::string("/ulx/ref/max"));

      //read vly
      vly_.data = internal::load_multi_array<3>(file, prefix + std::string("/vly/data"));
      vly_.np = internal::hdf5_read_int(file, prefix + std::string("/vly/np"));
      vly_.ns = internal::hdf5_read_int(file, prefix + std::string("/vly/ns"));
      vly_.section_edges = internal::load_multi_iarray<1>(file, prefix + std::string("/vly/section_edges"));

      //read ref_vly
      ref_vly_.data = internal::load_multi_array<2>(file, prefix + std::string("/vly/ref/data"));
      ref_vly_.max = internal::load_multi_array<1>(file, prefix + std::string("/vly/ref/max"));

      H5Fclose(file);
    }

    /**
      * Return number of basis functions
      * @return  number of basis functions
      */
    int dim() const { return dim_; }

    double sl(int l) const throw(std::runtime_error) {
      assert(l >= 0 && l < dim());
      return static_cast<double>(sl_(l));
    }

  private:
    double Lambda_;
    int dim_;
    std::string statistics_;
    internal::multi_array<double, 1> sl_;
    func ulx_;
    func vly_;
    ref ref_ulx_;
    ref ref_vly_;
  };

};
