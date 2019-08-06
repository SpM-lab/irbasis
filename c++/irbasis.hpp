#pragma once

#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <complex>
#include <hdf5.h>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/bessel.hpp>

//debug
//#include <chrono>
//
//#ifdef IRBASIS_USE_EIGEN3
//#include <Eigen/Core>
//#endif

namespace irbasis {
namespace mp = boost::multiprecision;
typedef mp::number<mp::cpp_dec_float<30> > mpf;

namespace internal {

// Simple implementation without meta programming...
template<typename T, int DIM>
class multi_array {

  template<typename T2, int DIM2>
  friend class multi_array;

public:
  multi_array() : owner_(true), p_data_(NULL), num_elements_(0) {
    for (int i = 0; i < DIM; ++i) {
      extents_[i] = 0;
    }
  }

  multi_array(int N1) : owner_(true), p_data_(new T[N1]), num_elements_(N1) {
    assert(DIM == 1);
    extents_[0] = N1;
  }

  multi_array(int N1, int N2) : owner_(true), p_data_(new T[N1 * N2]), num_elements_(N1 * N2) {
    assert(DIM == 2);
    extents_[0] = N1;
    extents_[1] = N2;
  }

  multi_array(int N1, int N2, int N3) : owner_(true), p_data_(new T[N1 * N2 * N3]), num_elements_(N1 * N2 * N3) {
    assert(DIM == 3);
    extents_[0] = N1;
    extents_[1] = N2;
    extents_[2] = N3;
  }

  multi_array(std::size_t *dims) {
    resize(dims);
  }

  multi_array(const multi_array<T, DIM> &other) : p_data_(NULL) {
    owner_ = true;
    resize(&other.extents_[0]);
    std::copy(other.origin(), other.origin() + other.num_elements(), origin());
  }

  ~multi_array() {
    if (this->owner_) {
      delete[] p_data_;
    }
  }

  multi_array<T, DIM> &operator=(const multi_array<T, DIM> &other) {
    if (!this->owner_) {
      throw std::logic_error("Error: assignment to a view is not supported.");
    }

    this->owner_ = other.owner_;
    for (int i = 0; i < DIM; ++i) {
      this->extents_[i] = other.extents_[i];
    }
    this->num_elements_ = other.num_elements_;

    // allocate memoery and copy data
    if (this->p_data_ != NULL) {
      delete[] this->p_data_;
      this->p_data_ = NULL;
    }
    this->p_data_ = new T[this->num_elements_];

    for (int i = 0; i < this->num_elements_; ++i) {
      *(this->p_data_ + i) = *(other.p_data_ + i);
    }

    return *this;
  }

  std::size_t extent(int i) const {
    assert(i >= 0);
    assert(i < DIM);
    return extents_[i];
  }

  const std::size_t* extents() const {
      return &(extents_[0]);
  }

  void resize(const std::size_t *const dims) {
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

  void fill(T value) {
    if (!owner_) {
      throw std::runtime_error("resize is not permitted for a view");
    }
    for (int i = 0; i < this->num_elements_; ++i) {
      *(this->p_data_ + i) = value;
    }
  }

  multi_array<T, DIM - 1> make_view(std::size_t most_left_index) const {
    multi_array<T, DIM - 1> view;
    view.owner_ = false;
    std::size_t new_size = 1;
    for (int i = 0; i < DIM - 1; ++i) {
      view.extents_[i] = this->extents_[i + 1];
      new_size *= view.extents_[i];
    }
    view.num_elements_ = new_size;
    view.p_data_ = p_data_ + most_left_index * new_size;

    return view;
  }

  multi_array<T, 2> make_matrix_view(std::size_t size1, std::size_t size2) const {
    multi_array<T, 2> view;
    view.owner_ = false;
    view.extents_[0] = size1;
    view.extents_[1] = size2;
    view.num_elements_ = num_elements_;
    view.p_data_ = p_data_;

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

  inline T &operator()(int i) {
    assert(DIM == 1);
    int idx = i;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

  inline const T &operator()(int i) const {
    assert(DIM == 1);
    int idx = i;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

  inline T &operator()(int i, int j) {
    assert(DIM == 2);
    int idx = extents_[1] * i + j;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

  inline const T &operator()(int i, int j) const {
    assert(DIM == 2);
    int idx = extents_[1] * i + j;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

  inline T &operator()(int i, int j, int k) {
    assert(DIM == 3);
    int idx = (i * extents_[1] + j) * extents_[2] + k;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

  inline const T &operator()(int i, int j, int k) const {
    assert(DIM == 3);
    int idx = (i * extents_[1] + j) * extents_[2] + k;
    assert(idx >= 0 && idx < num_elements());
    return *(p_data_ + idx);
  }

private:
  bool owner_;
  T *p_data_;
  std::size_t num_elements_;
  std::size_t extents_[DIM];
};

template<typename T>
void multiply(const multi_array<T, 2> &A, const multi_array<T, 2> &B, multi_array<T, 2> &AB) {
  std::size_t N1 = A.extent(0);
  std::size_t N2 = A.extent(1);
  std::size_t N3 = B.extent(1);

  assert(B.extent(0) == N2);

  if (AB.extent(0) != N1 || AB.extent(1) != N3) {
    std::size_t dims[2];
    dims[0] = N1;
    dims[1] = N3;
    AB.resize(dims);
  }

  AB.fill(0);
  for (int i = 0; i < N1; ++i) {
    for (int k = 0; k < N2; ++k) {
      for (int j = 0; j < N3; ++j) {
        AB(i, j) += A(i, k) * B(k, j);
      }
    }
  }

}

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

typedef struct {
  double re;   /*real part*/
  double im;   /*imaginary part*/
} complex_t;

template<>
inline
hid_t
get_native_type<std::complex<double> >() {
  hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
  H5Tinsert(complex_id, "r", HOFFSET(complex_t, re), H5T_NATIVE_DOUBLE);
  H5Tinsert(complex_id, "i", HOFFSET(complex_t, im), H5T_NATIVE_DOUBLE);
  return complex_id;
}

// read a scalar
template<typename T>
T hdf5_read_scalar(hid_t &file, const std::string &name) {
  hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("Failed to load dataset" + name);
  }
  T data;
  H5Dread(dataset, get_native_type<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, &data);
  H5Dclose(dataset);
  return data;
}

// read array of double
template<int DIM>
void hdf5_read_double_array(hid_t &file, const std::string &name, std::vector<std::size_t> &extents,
                            std::vector<double> &data) {
  hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("Failed to load dataset" + name);
  }
  hid_t space = H5Dget_space(dataset);
  std::vector<hsize_t> dims(DIM);
  int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
  assert(n_dims == DIM);
  std::size_t tot_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
  data.resize(tot_size);
  extents.resize(DIM);
  for (int i = 0; i < DIM; ++i) {
    extents[i] = static_cast<std::size_t>(dims[i]);
  }
  H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);
  H5Dclose(dataset);
}

template<int DIM>
void hdf5_read_mpf_array(hid_t &file, const std::string &name, std::vector<std::size_t> &extents,
                            std::vector<mpf> &data) {
  std::vector<double> data_tmp;
  hdf5_read_double_array<DIM>(file, name, extents, data_tmp);
  data.resize(data_tmp.size());
  for (int i = 0; i< data.size(); ++i) {
    data[i] = data_tmp[i];
  }

  hdf5_read_double_array<DIM>(file, name+"_corr", extents, data_tmp);
  for (int i = 0; i< data.size(); ++i) {
    data[i] += data_tmp[i];
  }

}

// read a multi_array
template<typename T, int DIM>
multi_array<T, DIM> load_multi_array(hid_t &file, const std::string &name) {
  hid_t dataset = H5Dopen2(file, name.c_str(), H5P_DEFAULT);
  if (dataset < 0) {
    throw std::runtime_error("Faild to open a dataset.");
  }
  hid_t space = H5Dget_space(dataset);
  std::vector<hsize_t> dims(DIM);
  int n_dims = H5Sget_simple_extent_dims(space, &dims[0], NULL);
  assert(n_dims == DIM);
  std::size_t
      tot_size = static_cast<std::size_t>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>()));
  std::vector<std::size_t> extents(DIM);
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

template<int DIM>
multi_array<mpf,DIM> load_mpf_multi_array(hid_t &file, const std::string &name) {
  multi_array<double,DIM> data = load_multi_array<double,DIM>(file, name);
  multi_array<double,DIM> data_corr = load_multi_array<double,DIM>(file, name + "_corr");

  multi_array<mpf,DIM> result;
  result.resize(data.extents());

  for (int i=0; i<result.num_elements(); ++i) {
      *(result.origin()+i) = *(data.origin()+i);
      *(result.origin()+i) += *(data_corr.origin()+i);
  }
  return result;
}

inline
std::size_t find_section(const multi_array<mpf, 1> &section_edges, double x) {
  std::size_t idx = std::upper_bound(
      section_edges.origin(),
      section_edges.origin() + section_edges.num_elements(),
      x) - section_edges.origin() - 1;

  return std::min(idx, section_edges.num_elements() - 2);
}

inline
void compute_legendre(double x, std::vector<double> &val) {
  for (int l = 0; l < val.size(); l++) {
    if (l == 0) {
      val[l] = 1;
    } else if (l == 1) {
      val[l] = x;
    } else {
      val[l] = ((2 * l - 1) * x * val[l - 1] - (l - 1) * val[l - 2]) / l;
    }
  }
}


inline
int even_odd_sign(const int l) {
  return (l % 2 == 0 ? 1 : -1);
}

inline
multi_array<std::complex<double>, 2>
compute_unl_tail(std::vector<double> &w_vec,
                 const std::string &statistics,
                 const multi_array<double, 2> &derive_x1,
                 const int n) {
  // n : target number to calculate
  int sign = statistics == "B" ? 1 : -1;
  std::size_t n_iw = w_vec.size();
  std::size_t Nl = derive_x1.extent(0);
  std::size_t num_deriv = derive_x1.extent(1);
  multi_array<std::complex<double>, 2> result(n_iw, Nl);
  result.fill(std::complex<double>(0.0, 0.0));
  if (n > 0)
    num_deriv -= n;
  multi_array<std::complex<double>, 2> coeffs_nm(n_iw, num_deriv);
  coeffs_nm.fill(std::complex<double>(0.0, 0.0));

  //coeffs_nm
  for (int i_iw = 0; i_iw < n_iw; i_iw++) {
    if (statistics == "B" && w_vec[i_iw] == 0) {
      continue;
    }
    std::complex<double> fact = std::complex<double>(0.0, 1.0 / w_vec[i_iw]);
    coeffs_nm(i_iw, 0) = fact;
    for (int m = 1; m < num_deriv; m++) {
      coeffs_nm(i_iw, m) = fact * coeffs_nm(i_iw, m - 1);
    }
  }

  //coeffs_lm ()
  multi_array<std::complex<double>, 2> coeffs_lm(Nl, num_deriv);
  coeffs_lm.fill(std::complex<double>(0.0, 0.0));
  for (int l = 0; l < Nl; l++) {
    for (int m = 0; m < num_deriv; m++) {
      coeffs_lm(l, m) = (1.0 - sign * even_odd_sign(l + m)) * derive_x1(l, m);
    }
  }

  for (int i = 0; i < n_iw; i++) {
    for (int k = 0; k < Nl; k++) {
      for (int j = 0; j < num_deriv; j++) {
        result(i, k) += coeffs_nm(i, j) * coeffs_lm(k, j);
      }
      result(i, k) *= -sign / sqrt(2.0);
    }
  }
  return result;
}

struct func {
  void load_from_h5(hid_t file, const std::string &prefix) {
    data = internal::load_multi_array<double, 3>(file, prefix + std::string("/data"));
    np = internal::hdf5_read_scalar<int>(file, prefix + std::string("/np"));
    ns = internal::hdf5_read_scalar<int>(file, prefix + std::string("/ns"));
    nl = data.extent(0);
    section_edges = internal::load_mpf_multi_array<1>(file, prefix + std::string("/section_edges"));

    std::size_t extents[3];
    extents[0] = ns;
    extents[1] = np;
    extents[2] = nl;
    data_for_vec.resize(&extents[0]);
    for (int l = 0; l < nl; ++l) {
      for (int s = 0; s < ns; ++s) {
        for (int p = 0; p < np; ++p) {
          data_for_vec(s, p, l) = data(l, s, p);
        }
      }
    }
  }
  multi_array<mpf, 1> section_edges;
  multi_array<double, 3> data; //(nl, ns, np)
  multi_array<double, 3> data_for_vec; //(ns, np, nl). Just a copy of data.
  int np;
  int ns;
  int nl;
};

struct ref {
  multi_array<double, 2> data;
  multi_array<double, 1> max;
};

}

class basis {
public:
  basis() {}

  basis(
      const std::string &file_name,
      const std::string &prefix = ""
  ) throw(std::runtime_error) {
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file < 0) {
      throw std::runtime_error("Failed to open " + file_name + "!");
    }

    //read info
    Lambda_ = internal::hdf5_read_scalar<double>(file, prefix + std::string("/info/Lambda"));
    dim_ = internal::hdf5_read_scalar<int>(file, prefix + std::string("/info/dim"));
    statistics_ = internal::hdf5_read_scalar<int>(file, prefix + std::string("/info/statistics")) == 0 ? "B" : "F";

    //read sl
    sl_ = internal::load_multi_array<double, 1>(file, prefix + std::string("/sl"));

    //read ulx
    ulx_.load_from_h5(file, prefix + "/ulx");

    //read ref_ulx
    ref_ulx_.data = internal::load_multi_array<double, 2>(file, prefix + std::string("/ulx/ref/data"));
    ref_ulx_.max = internal::load_multi_array<double, 1>(file, prefix + std::string("/ulx/ref/max"));

    //read vly
    vly_.load_from_h5(file, prefix + "/vly");

    //read ref_vly
    ref_vly_.data = internal::load_multi_array<double, 2>(file, prefix + std::string("/vly/ref/data"));
    ref_vly_.max = internal::load_multi_array<double, 1>(file, prefix + std::string("/vly/ref/max"));

    H5Fclose(file);

    int np = ulx_.np;
    {
      std::vector<double> coeffs(np);
      for (int p=0; p<np; ++p) {
        //Conversion between \tilde{P}_l and P_l
        coeffs[p] = std::sqrt(p + 0.5);
      }

      std::size_t dims[2];
      dims[0] = np;
      dims[1] = np;
      deriv_mat_.resize(dims);
      deriv_mat_.fill(0.0);
      for (int l=0; l<np; ++l) {
        for (int m=l-1; m>=0; m-=2) {
          deriv_mat_(m, l) = (1/coeffs[m]) * (2*m + 1) * coeffs[l];
        }
      }
    }

    norm_coeff_.resize(np);
    for (int p=0; p<np; ++p) {
      norm_coeff_[p] = std::sqrt(p + 0.5);
    }

  }

  /**
    * Return number of basis functions
    * @return  number of basis functions
    */
  int dim() const { return dim_; }

  /**
    * Return Lambda
    * @return  Lambda
    */
  double Lambda() const { return Lambda_; }

  /**
    * Return statistics
    * @return  string representing statistics "F" or "B"
    */
  std::string statistics() const { return statistics_; }


  double sl(int l) const throw(std::runtime_error) {
    assert(l >= 0 && l < dim());
    return sl_(l);
  }

  double ulx(int l, double x) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(x) > 1) {
        throw std::runtime_error("x must be in [-1,1]!");
    }

    if (x >= 0) {
      return eval(x, ulx_.data.make_view(l), ulx_.section_edges);
    } else {
      return eval(-x, ulx_.data.make_view(l), ulx_.section_edges) * even_odd_sign(l);
    }
  }

  std::vector<std::vector<double> > check_ulx() const {
    double ulx_max = ref_ulx_.max(2);
    std::vector<std::vector<double> > ref_data(ref_ulx_.data.extent(0));
    int count = 0;
    for (int i = 0; i < ref_ulx_.data.extent(0); i++) {
      if (ref_ulx_.data(i, 2) == 0) {
        ref_data[i].push_back(ref_ulx_.data(i, 0));
        ref_data[i].push_back(ref_ulx_.data(i, 1));
        ref_data[i].push_back(
            fabs(ulx(ref_ulx_.data(i, 0) - 1, ref_ulx_.data(i, 1)) - ref_ulx_.data(i, 3)) / ulx_max);
        count++;
      }
    }
    ref_data.resize(count);
    return ref_data;
  }

  std::vector<std::vector<double> > check_vly() const {
    double vly_max = ref_vly_.max(2);
    std::vector<std::vector<double> > ref_data(ref_vly_.data.extent(0));
    int count = 0;
    for (int i = 0; i < ref_vly_.data.extent(0); i++) {
      if (ref_vly_.data(i, 2) == 0) {
        ref_data[i].push_back(ref_vly_.data(i, 0));
        ref_data[i].push_back(ref_vly_.data(i, 1));
        ref_data[i].push_back(
            fabs(vly(ref_vly_.data(i, 0) - 1, ref_vly_.data(i, 1)) - ref_vly_.data(i, 3)) / vly_max);
        count++;
      }
    }
    ref_data.resize(count);
    return ref_data;
  }

  double d_ulx(int l, double x, std::size_t order) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(x) > 1) {
        throw std::runtime_error("x must be in [-1,1]!");
    }

    if (x >= 0) {
      return eval_derivative(x, order, ulx_.data.make_view(l), ulx_.section_edges);
    } else {
      return eval_derivative(-x, order, ulx_.data.make_view(l), ulx_.section_edges) * even_odd_sign(l + order);
    }
  }

  double vly(int l, double y) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(y) > 1) {
        throw std::runtime_error("y must be in [-1,1]!");
    }
    if (y >= 0) {
      return eval(y, vly_.data.make_view(l), vly_.section_edges);
    } else {
      return eval(-y, vly_.data.make_view(l), vly_.section_edges) * even_odd_sign(l);
    }
  }

  double d_vly(int l, double y, std::size_t order) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(y) > 1) {
        throw std::runtime_error("y must be in [-1,1]!");
    }
    if (y >= 0) {
      return eval_derivative(y, order, vly_.data.make_view(l), vly_.section_edges);
    } else {
      return eval_derivative(-y, order, vly_.data.make_view(l), vly_.section_edges) * even_odd_sign(l + order);
    }
  }

  double get_ref_ulx(std::size_t order) const {
    double ref_data;
    for (int i = 0; i < ref_ulx_.data.extent(0); i++) {
      if (ref_ulx_.data(i, 2) == order) {
        ref_data = ref_ulx_.data(i, 3);
      }
    }
    return ref_data;
  }

  int num_sections_x() const {
    return ulx_.data.extent(1);
  }

  double section_edge_x(std::size_t index) const {
    assert(index >= 0 && index <= num_sections_x());
    return static_cast<double>(ulx_.section_edges(index));
  }

  int num_sections_y() const {
    return vly_.data.extent(1);
  }

  std::vector<std::vector<std::complex<double> > > compute_unl(long long n) const {
      std::vector<long long> n_vec;
      n_vec.push_back(n);
      return compute_unl(n_vec);
  }

  std::vector<std::vector<std::complex<double> > > compute_unl(const std::vector<long long> &n) const {
    using namespace internal;

    typedef std::complex<double> dcomplex;
    typedef std::complex<mpf> mcomplex; // may be illegal to instantiate std::complex with mpf?

    mpf mpi = boost::math::constants::pi<mpf>();

    int num_n = n.size();
    std::complex<double> J = std::complex<double>(0.0, 1.0);

    std::vector<mpf> o_vec(n.size());
    if (this->statistics_ == "F") {
      for (int i = 0; i < num_n; i++) {
        o_vec[i] = (mpf(2) * n[i] + 1);
      }
    } else {
      for (int i = 0; i < num_n; i++) {
        o_vec[i] = (mpf(2) * n[i]);
      }
    }

    //w_vec = 0.5 * pi * o_vec
    std::vector<mpf> w_vec(o_vec);
    std::transform(w_vec.begin(), w_vec.end(), w_vec.begin(), std::bind1st(std::multiplies<mpf>(), mpi/2));
    std::vector<double> w_vec_f(num_n);
    for (int n=0; n<num_n; ++n) {
      w_vec_f[n] = static_cast<double>(w_vec[n]);
    }

    std::size_t num_deriv = this->ulx_.data.extent(2);

    //Compute tail
    std::vector<std::vector<int> > replaced_with_tail(num_n, std::vector<int>(this->dim_, 0));
    multi_array<double, 2> deriv_x1(this->dim_, num_deriv);
    deriv_x1.fill(0.0);
    std::vector<double> d_ulx_result;
    for (int l = 0; l < dim_; ++l) {
      for (int p = 0; p < num_deriv; ++p) {
        deriv_x1(l, p) = d_ulx(l, 1.0, p);
      }
    }

    multi_array<std::complex<double>, 2> unl_tail = compute_unl_tail(w_vec_f, statistics_, deriv_x1, -1);
    multi_array<std::complex<double>, 2> unl_tail_without_last_two = compute_unl_tail(w_vec_f, statistics_, deriv_x1, 2);

    for (int i = 0; i < num_n; i++) {
      if (statistics_ == "B" && n[i] == 0)
        continue;
      for (int l = 0; l < dim_; ++l) {
        if (std::abs((unl_tail(i, l) - unl_tail_without_last_two(i, l)) / unl_tail(i, l)) < 1e-10) {
          replaced_with_tail[i][l] = 1;
        }
      }
    }

    multi_array<std::complex<mpf>,2> tilde_unl = compute_tilde_unl_fast(w_vec);

    std::vector<std::vector<std::complex<double> > > result_vec(num_n, std::vector<std::complex<double> >(dim_, 0));
    int sign_shift = statistics_ == "F" ? 1 : 0;
    for (int l = 0; l < dim_; ++l) {
      if ((l + sign_shift)%2 == 1) {
        for (int n=0; n<num_n; ++n) {
          result_vec[n][l] = std::complex<double>(0, 2*static_cast<double>(tilde_unl(n, l).imag()));
        }
      } else {
        for (int n=0; n<num_n; ++n) {
          result_vec[n][l] = 2 * static_cast<double>(tilde_unl(n, l).real());
        }
      }
    }


    //Overwrite by tail
    for (int i = 0; i < num_n; i++) {
      for (int l = 0; l < dim_; l++) {
        if (replaced_with_tail[i][l] == 1) {
          result_vec[i][l] = unl_tail(i, l);
        }
      }
    }
    return result_vec;
  }

  double section_edge_y(std::size_t index) const {
    assert(index >= 0 && index <= num_sections_y());
    return static_cast<double>(vly_.section_edges(index));
  }


protected:
  double Lambda_;
  int dim_;
  std::string statistics_;
  internal::multi_array<double, 1> sl_;
  internal::func ulx_;
  internal::func vly_;
  internal::ref ref_ulx_;
  internal::ref ref_vly_;
  internal::multi_array<double, 2> deriv_mat_;
  std::vector<double> norm_coeff_;

private:
  // Evaluate the value of function at given x
  double eval(double x, const internal::multi_array<double, 2> &data, const internal::multi_array<mpf, 1> &section_edges) const {
    std::size_t section_idx = find_section(section_edges, x);
    return eval_impl(x, section_edges(section_idx), section_edges(section_idx+1), data.make_view(section_idx));
  };

  double eval_impl(double x, mpf x_s, mpf x_sp, const internal::multi_array<double, 1> &coeffs) const {
    mpf dx = x_sp - x_s;
    mpf tilde_x = (2*x - x_sp - x_s)/dx;

    std::vector<double> leg_vals(coeffs.extent(0));
    internal::compute_legendre(static_cast<double>(tilde_x), leg_vals);
    double eval_result = 0.0;
    for (int p=0; p<leg_vals.size(); ++p) {
      eval_result += leg_vals[p] * norm_coeff_[p] * coeffs(p);
    }
    return eval_result * std::sqrt(2/static_cast<double>(dx));
  }

  void
  differentiate_coeff(internal::multi_array<double, 1> &coeffs, std::size_t order) const {
    std::size_t k = coeffs.num_elements();

    internal::multi_array<double, 2> coeffs_deriv(coeffs.make_matrix_view(k, 1));
    internal::multi_array<double, 2> tmp(coeffs_deriv);

    for (int i=0; i<order; ++i) {
      return internal::multiply(deriv_mat_, coeffs_deriv, tmp);
      std::swap(coeffs_deriv, tmp);
    }

    for (int p=0; p<k; ++k) {
      coeffs(p) = coeffs_deriv(p);
    }
  }

  double eval_derivative(double x,
                                std::size_t order,
                                const internal::multi_array<double, 2> &data,
                                const internal::multi_array<mpf, 1> &section_edges,
                                int section = -1) const {
    using namespace internal;
    std::size_t section_idx = section >= 0 ? section : find_section(section_edges, x);

    multi_array<double, 1> coeffs_deriv(data.make_view(section_idx));
    differentiate_coeff(coeffs_deriv, order);
    double dx = static_cast<double>(section_edges(section_idx+1) - section_edges(section_idx));
    return eval_impl(x, section_edges(section_idx), section_edges(section_idx+1), coeffs_deriv) * std::pow(2/dx, order);
  }

  internal::multi_array<std::complex<mpf>,2> compute_tilde_unl_fast(const std::vector<mpf>& w_vec) const {
    int num_n = w_vec.size();
    int np = ulx_.np;

    typedef std::complex<mpf> mcomplex;
    typedef std::complex<double> dcomplex;

    internal::multi_array<mcomplex,2> tilde_unl(num_n, dim_);
    tilde_unl.fill(mcomplex(0,0));

    internal::multi_array<double,2> tmp_lp(dim_, np);
    internal::multi_array<dcomplex,2> tmp_np(num_n, np);
    std::vector<dcomplex> exp_n(num_n);

    for (int s=0; s<num_sections_x(); ++s) {
      mpf xs = ulx_.section_edges(s);
      mpf xsp = ulx_.section_edges(s+1);
      mpf dx = xsp - xs;
      mpf xmid = (xsp + xs)/2;

      // tmp_lp: lp
      {
        // Normalization factor
        double coeff_tmp = std::sqrt(static_cast<double>(dx))/2;
        for (int l=0; l<dim_; ++l) {
          for (int p=0; p<np; ++p) {
            tmp_lp(l, p) = ulx_.data(l, s, p) * coeff_tmp * norm_coeff_[p];
          }
        }
      }

      // tmp_np: np
      for (int n=0; n<num_n; ++n) {
        mpf phase = w_vec[n] * (xmid+1);
        exp_n[n] = std::complex<double>(
              static_cast<double>(boost::multiprecision::cos(phase)),
              static_cast<double>(boost::multiprecision::sin(phase))
            );
      }
      for (int n=0; n<num_n; ++n) {
        double w_tmp = static_cast<double>(dx * w_vec[n]/2);
        dcomplex phase_p(0, 0);
        for (int p = 0; p < np; ++p) {
          tmp_np(n, p) = 2.0 * phase_p * boost::math::sph_bessel(p, w_tmp) * exp_n[n];
          phase_p *= dcomplex(0, 1);
        }
      }

      for (int n=0; n<num_n; ++n) {
        for (int l=0; l<dim_; ++l) {
          for (int p=0; p<np; ++p) {
             tilde_unl(n, l) += tmp_np(n, p) * tmp_lp(l, p);
          }
        }
      }
    }
    return tilde_unl;
  }
};

inline
basis load(const std::string &statistics, double Lambda, const std::string &file_name = "./irbasis.h5") {
  std::stringstream ss;
  ss << std::fixed;
  ss << std::setprecision(1);
  ss << Lambda;
  std::string prefix;
  if (statistics == "F") {
    prefix = "basis_f-mp-Lambda" + ss.str() + "_np8";
  } else if (statistics == "B") {
    prefix = "basis_b-mp-Lambda" + ss.str() + "_np8";
  } else {
    throw std::runtime_error("Unsupported statistics " + statistics);
  }

  return basis(file_name, prefix);
}
};


