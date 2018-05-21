#pragma once

#include <iostream>
#include <complex>
#include <cmath>
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

      template<typename T2, int DIM2>
      friend class multi_array;

    public:
      multi_array() : owner_(true), p_data_(NULL), num_elements_(0) {
      }

      multi_array(int N1) : owner_(true), p_data_(new T[N1]), num_elements_(N1) {
        assert(DIM == 1);
        extents_[0] = N1;
      }

      multi_array(int N1, int N2) : owner_(true), p_data_(new T[N1*N2]), num_elements_(N1*N2) {
        assert(DIM == 2);
        extents_[0] = N1;
        extents_[1] = N2;
      }

      multi_array(int N1, int N2, int N3) : owner_(true), p_data_(new T[N1*N2*N3]), num_elements_(N1*N2*N3) {
        assert(DIM == 3);
        extents_[0] = N1;
        extents_[1] = N2;
        extents_[2] = N3;
      }

      multi_array(std::size_t *dims) {
        resize(dims);
      }

      multi_array(const multi_array<T,DIM>& other) : p_data_(NULL) {
          *this = other;
      }

      ~multi_array() {
        if (this->owner_) {
            delete[] p_data_;
        }
      }

      multi_array<T,DIM>& operator=(const multi_array<T,DIM>& other) {
          this->owner_ = other.owner_;
          for (int i = 0; i < DIM; ++i) {
              this->extents_[i] = other.extents_[i];
          }
          this->num_elements_ = other.num_elements_;

          if (this->p_data_ != NULL) {
              delete[] this->p_data_;
              this->p_data_ = NULL;
          }

          if (other.owner_) {
              // allocate memoery and copy data
              if (this->p_data_ == NULL) {
                  this->p_data_ = new T[this->num_elements_];
              }
              for (int i=0; i<this->num_elements_; ++i) {
                  *(this->p_data_+i) = *(other.p_data_+i);
              }
          } else {
              // point to the same data
              this->p_data_ = other.p_data_;
          }

          return *this;
      }

      std::size_t extent(int i) const {
        assert(i >= 0);
        assert(i < DIM);
        return extents_[i];
      }

      void resize(std::size_t *dims) {
        if (!owner_) {
            throw std::runtime_error("resize is not permitted for a view");
        }

        std::size_t tot_size = std::accumulate(dims, dims + DIM, 1, std::multiplies<std::size_t>());
        delete[] p_data_;
        p_data_ = new T[tot_size];
        num_elements_ = tot_size;
        for (int i = 0; i < DIM; ++i) {
          extents_[i] = dims[i];
        }
      }

      multi_array<T,DIM-1> make_view(std::size_t most_left_index) const {
          multi_array<T,DIM-1> view;
          view.owner_ = false;
          std::size_t new_size = 1;
          for (int i=0; i<DIM-1; ++i) {
              view.extents_[i] = this->extents_[i+1];
              new_size *= view.extents_[i];
          }
          view.num_elements_ = new_size;
          view.p_data_ = p_data_ + most_left_index * new_size;

          return view;
      }

      std::size_t num_elements() const {
        return num_elements_;
      }

      bool is_view() const {
          return !owner_;
      }

      T *origin() const {
        return p_data_;
      }

      T &operator()(int i) {
        assert(DIM == 1);
        int idx = i;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

      const T &operator()(int i) const {
        assert(DIM == 1);
        int idx = i;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

      T &operator()(int i, int j) {
        assert(DIM == 2);
        int idx = extents_[1] * i + j;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

      const T &operator()(int i, int j) const {
        assert(DIM == 2);
        int idx = extents_[1] * i + j;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

      T &operator()(int i, int j, int k) {
        assert(DIM == 3);
        int idx = (i * extents_[1] + j) * extents_[2] + k;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

      const T &operator()(int i, int j, int k) const {
        assert(DIM == 3);
        int idx = (i * extents_[1] + j) * extents_[2] + k;
        assert(idx >= 0 && idx < num_elements());
        return *(p_data_+idx);
      }

    private:
      bool owner_;
      T* p_data_;
      std::size_t num_elements_;
      std::size_t extents_[DIM];
    };

    // https://www.physics.ohio-state.edu/~wilkins/computing/HDF/hdf5tutorial/examples/C/h5_rdwt.c
    // https://support.hdfgroup.org/ftp/HDF5/current/src/unpacked/examples/h5_read.c
    template<typename T> hid_t get_native_type();

    template<>
    inline
    hid_t
    get_native_type<double>() {
        return H5T_NATIVE_DOUBLE;
    }

    template<>
    inline
    hid_t
    get_native_type<int>() {
        return H5T_NATIVE_INT;
    }

    // read a scalar
    template<typename T>
    T hdf5_read_scalar(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      T data;
      H5Dread(dataset, get_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
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

    // read a multi_array
    template<typename T, int DIM>
    multi_array<T, DIM> load_multi_array(hid_t &file, const std::string &name) {
      hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
      if (dataset < 0) {
          throw std::runtime_error("Faild to open a dataset.");
      }
      hid_t space = H5Dget_space(dataset);
      std::vector <hsize_t> dims(DIM);
      int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
      assert(n_dims == DIM);
      std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
      std::vector <std::size_t> extents(DIM);
      for (int i = 0; i < DIM; ++i) {
        extents[i] = static_cast<std::size_t>(dims[i]);
      }
      multi_array<T, DIM> a;
      a.resize(&extents[0]);
      H5Dread(dataset, get_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, a.origin());
      std::vector<T> data(tot_size);
      H5Dread(dataset, get_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
      H5Dclose(dataset);
      return a;
    }

    inline
    std::size_t find_section(const multi_array<double,1> &section_edges, double x) {
        std::size_t idx = std::upper_bound(
                             section_edges.origin(),
                             section_edges.origin() + section_edges.num_elements(),
                             x) - section_edges.origin() - 1;

        return std::min(idx, section_edges.num_elements()-2);
    }

    inline
    double interpolate_impl(double dx, const multi_array<double,1>& coeffs) {
        double value = 0.0;
        double dx_power = 1.0;
        std::size_t N = coeffs.num_elements();
        for (int p=0; p < N; ++p) {
            value += dx_power * coeffs(p);
            dx_power *= dx;
        }
        return value;
    }

    inline
    double interpolate(double x, const multi_array<double,2> &_data, const multi_array<double,1> &section_edges) {
        std::size_t section_idx = find_section(section_edges, x);
        return interpolate_impl(x - section_edges(section_idx), _data.make_view(section_idx));
    }

    inline
    int even_odd_sign(const int l){
      return (l%2==0 ? 1 : -1);
    }

  }

  struct func {
    internal::multi_array<double, 1> section_edges;
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
      Lambda_ = internal::hdf5_read_scalar<double>(file, prefix + std::string("/info/Lambda"));
      dim_ = internal::hdf5_read_scalar<int>(file, prefix + std::string("/info/dim"));
      statistics_ = internal::hdf5_read_scalar<int>(file, prefix + std::string("/info/statistics")) == 0 ? "B" : "F";

      //read sl
      sl_ = internal::load_multi_array<double,1>(file, prefix + std::string("/sl"));

      //read ulx
      ulx_.data = internal::load_multi_array<double,3>(file, prefix + std::string("/ulx/data"));
      ulx_.np = internal::hdf5_read_scalar<int>(file, prefix + std::string("/ulx/np"));
      ulx_.ns = internal::hdf5_read_scalar<int>(file, prefix + std::string("/ulx/ns"));
      ulx_.section_edges = internal::load_multi_array<double,1>(file, prefix + std::string("/ulx/section_edges"));

      //read ref_ulx
      ref_ulx_.data = internal::load_multi_array<double,2>(file, prefix + std::string("/ulx/ref/data"));
      ref_ulx_.max = internal::load_multi_array<double,1>(file, prefix + std::string("/ulx/ref/max"));

      //read vly
      vly_.data = internal::load_multi_array<double,3>(file, prefix + std::string("/vly/data"));
      vly_.np = internal::hdf5_read_scalar<int>(file, prefix + std::string("/vly/np"));
      vly_.ns = internal::hdf5_read_scalar<int>(file, prefix + std::string("/vly/ns"));
      vly_.section_edges = internal::load_multi_array<double,1>(file, prefix + std::string("/vly/section_edges"));

      //read ref_vly
      ref_vly_.data = internal::load_multi_array<double,2>(file, prefix + std::string("/vly/ref/data"));
      ref_vly_.max = internal::load_multi_array<double,1>(file, prefix + std::string("/vly/ref/max"));

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

    double ulx(int l, double x) const {
      using namespace internal;
      if(x >= 0) {
          return interpolate(x, ulx_.data.make_view(l), ulx_.section_edges);
      } else {
          return interpolate(-x, ulx_.data.make_view(l), ulx_.section_edges) * even_odd_sign(l);
      }
    }

    std::vector < std::vector <double> > check_ulx() const {
      double ulx_max = ref_ulx_.max(2);
      std::vector < std::vector<double> > ref_data(ref_ulx_.data.extent(0));
      int count = 0;
      for (int i=0; i<ref_ulx_.data.extent(0); i++) {
          if (ref_ulx_.data(i, 2) == 0) {
            ref_data[i].push_back(ref_ulx_.data(i, 0));
            ref_data[i].push_back(ref_ulx_.data(i, 1));
            ref_data[i].push_back(fabs(ulx(ref_ulx_.data(i, 0)-1, ref_ulx_.data(i, 1)) - ref_ulx_.data(i, 3))/ulx_max);
            count++;
          }
      }
      ref_data.resize(count);
      return ref_data;
    }

    std::vector < std::vector <double> > check_vly() const {
      double vly_max = ref_vly_.max(2);
      std::vector < std::vector<double> > ref_data(ref_vly_.data.extent(0));
      int count = 0;
      for (int i=0; i<ref_vly_.data.extent(0); i++) {
        if (ref_vly_.data(i, 2) == 0) {
          ref_data[i].push_back(ref_vly_.data(i, 0));
          ref_data[i].push_back(ref_vly_.data(i, 1));
          ref_data[i].push_back(fabs(vly(ref_vly_.data(i, 0)-1, ref_vly_.data(i, 1)) - ref_vly_.data(i, 3))/vly_max);
          count++;
        }
      }
      ref_data.resize(count);
      return ref_data;
    }

    double vly (int l, double y) const {
      using namespace internal;
      if(y >= 0) {
          return interpolate(y, vly_.data.make_view(l), vly_.section_edges);
      } else {
          return interpolate(-y, vly_.data.make_view(l), vly_.section_edges) * even_odd_sign(l);
      }
    }

    int num_sections_x() const {
      return ulx_.data.extent(1);
    }

    int num_sections_y() const {
      return vly_.data.extent(1);
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

    /*
    internal::multi_array<double, 2> get_l_data(int l, const internal::multi_array<double, 3> &_data){
      internal::multi_array<double, 2> a(_data.extent(1), _data.extent(2));
      for (int i=0; i<_data.extent(1); i++){
        for (int j=0; j<_data.extent(2); j++) a(i,j) = _data(l, i, j);
      }
      return a;
    }
    */



  };

};
