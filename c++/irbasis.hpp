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

//debug
//#include <chrono>
//
//#ifdef IRBASIS_USE_EIGEN3
//#include <Eigen/Core>
//#endif

namespace irbasis {
namespace internal {

inline
std::vector<std::pair<double,double> >
gauss_legendre_nodes(int num_nodes) {


    if (num_nodes == 24) {
        std::vector<std::pair<double,double> > nodes(24);

        nodes[0    ] = std::make_pair<double>(0.9951872199970213106468009, 0.01234122979998720018302016);
        nodes[1    ] = std::make_pair<double>(-0.9951872199970213106468009, 0.01234122979998720018302016);
        nodes[2    ] = std::make_pair<double>(0.9747285559713094738043537, 0.02853138862893366337059042);
        nodes[3    ] = std::make_pair<double>(-0.9747285559713094738043537, 0.02853138862893366337059042);
        nodes[4    ] = std::make_pair<double>(0.9382745520027327978951348, 0.04427743881741980774835454);
        nodes[5    ] = std::make_pair<double>(-0.9382745520027327978951348, 0.04427743881741980774835454);
        nodes[6    ] = std::make_pair<double>(0.8864155270044010714869387, 0.05929858491543678333801637);
        nodes[7    ] = std::make_pair<double>(-0.8864155270044010714869387, 0.05929858491543678333801637);
        nodes[8    ] = std::make_pair<double>(0.8200019859739029470802052, 0.07334648141108029983925576);
        nodes[9    ] = std::make_pair<double>(-0.8200019859739029470802052, 0.07334648141108029983925576);
        nodes[10   ] = std::make_pair<double>(0.7401241915785543579175965, 0.08619016153195327434310968);
        nodes[11   ] = std::make_pair<double>(-0.7401241915785543579175965, 0.08619016153195327434310968);
        nodes[12   ] = std::make_pair<double>(0.6480936519369755455244331, 0.09761865210411388438238589);
        nodes[13   ] = std::make_pair<double>(-0.6480936519369755455244331, 0.09761865210411388438238589);
        nodes[14   ] = std::make_pair<double>(0.545421471388839562699502, 0.1074442701159656343712356);
        nodes[15   ] = std::make_pair<double>(-0.545421471388839562699502, 0.1074442701159656343712356);
        nodes[16   ] = std::make_pair<double>(0.4337935076260451272567309, 0.1155056680537255991980672);
        nodes[17   ] = std::make_pair<double>(-0.4337935076260451272567309, 0.1155056680537255991980672);
        nodes[18   ] = std::make_pair<double>(0.3150426796961633968408023, 0.121670472927803391405277);
        nodes[19   ] = std::make_pair<double>(-0.3150426796961633968408023, 0.121670472927803391405277);
        nodes[20   ] = std::make_pair<double>(0.1911188674736163106704367, 0.1258374563468283025002847);
        nodes[21   ] = std::make_pair<double>(-0.1911188674736163106704367, 0.1258374563468283025002847);
        nodes[22   ] = std::make_pair<double>(0.06405689286260562997910029, 0.1279381953467521593204026);
        nodes[23   ] = std::make_pair<double>(-0.06405689286260562997910029, 0.1279381953467521593204026);

        return nodes;
    }


    if (num_nodes == 48) {
        std::vector<std::pair<double,double> > nodes(48);

        nodes[0    ] = std::make_pair<double>(0.998771007252426068490081, 0.003153346052305838493473589);
        nodes[1    ] = std::make_pair<double>(-0.998771007252426068490081, 0.003153346052305838493473589);
        nodes[2    ] = std::make_pair<double>(0.9935301722663507639765612, 0.007327553901276262493524882);
        nodes[3    ] = std::make_pair<double>(-0.9935301722663507639765612, 0.007327553901276262493524882);
        nodes[4    ] = std::make_pair<double>(0.9841245837228268511509555, 0.01147723457923453989348861);
        nodes[5    ] = std::make_pair<double>(-0.9841245837228268511509555, 0.01147723457923453989348861);
        nodes[6    ] = std::make_pair<double>(0.9705915925462472726437113, 0.01557931572294384871268935);
        nodes[7    ] = std::make_pair<double>(-0.9705915925462472726437113, 0.01557931572294384871268935);
        nodes[8    ] = std::make_pair<double>(0.9529877031604309101098238, 0.01961616045735552898987564);
        nodes[9    ] = std::make_pair<double>(-0.9529877031604309101098238, 0.01961616045735552898987564);
        nodes[10   ] = std::make_pair<double>(0.9313866907065543321309065, 0.02357076083932438045898117);
        nodes[11   ] = std::make_pair<double>(-0.9313866907065543321309065, 0.02357076083932438045898117);
        nodes[12   ] = std::make_pair<double>(0.905879136715569632798406, 0.02742650970835694770877389);
        nodes[13   ] = std::make_pair<double>(-0.905879136715569632798406, 0.02742650970835694770877389);
        nodes[14   ] = std::make_pair<double>(0.8765720202742478539548188, 0.03116722783279808964285174);
        nodes[15   ] = std::make_pair<double>(-0.8765720202742478539548188, 0.03116722783279808964285174);
        nodes[16   ] = std::make_pair<double>(0.8435882616243934872812815, 0.03477722256477044221467665);
        nodes[17   ] = std::make_pair<double>(-0.8435882616243934872812815, 0.03477722256477044221467665);
        nodes[18   ] = std::make_pair<double>(0.8070662040294426242681425, 0.03824135106583070875529984);
        nodes[19   ] = std::make_pair<double>(-0.8070662040294426242681425, 0.03824135106583070875529984);
        nodes[20   ] = std::make_pair<double>(0.7671590325157403578160142, 0.04154508294346474783775847);
        nodes[21   ] = std::make_pair<double>(-0.7671590325157403578160142, 0.04154508294346474783775847);
        nodes[22   ] = std::make_pair<double>(0.7240341309238146338955744, 0.04467456085669428006434956);
        nodes[23   ] = std::make_pair<double>(-0.7240341309238146338955744, 0.04467456085669428006434956);
        nodes[24   ] = std::make_pair<double>(0.6778723796326638906251105, 0.04761665849249047816060809);
        nodes[25   ] = std::make_pair<double>(-0.6778723796326638906251105, 0.04761665849249047816060809);
        nodes[26   ] = std::make_pair<double>(0.6288673967765135985885649, 0.05035903555385447261105725);
        nodes[27   ] = std::make_pair<double>(-0.6288673967765135985885649, 0.05035903555385447261105725);
        nodes[28   ] = std::make_pair<double>(0.5772247260839726834547037, 0.0528901894851936671404502);
        nodes[29   ] = std::make_pair<double>(-0.5772247260839726834547037, 0.0528901894851936671404502);
        nodes[30   ] = std::make_pair<double>(0.5231609747222329964699838, 0.05519950369998416483952042);
        nodes[31   ] = std::make_pair<double>(-0.5231609747222329964699838, 0.05519950369998416483952042);
        nodes[32   ] = std::make_pair<double>(0.4669029047509584140485117, 0.05727729210040321400354557);
        nodes[33   ] = std::make_pair<double>(-0.4669029047509584140485117, 0.05727729210040321400354557);
        nodes[34   ] = std::make_pair<double>(0.4086864819907167212242882, 0.05911483969839563534787175);
        nodes[35   ] = std::make_pair<double>(-0.4086864819907167212242882, 0.05911483969839563534787175);
        nodes[36   ] = std::make_pair<double>(0.348755886292160754980074, 0.06070443916589388089199986);
        nodes[37   ] = std::make_pair<double>(-0.348755886292160754980074, 0.06070443916589388089199986);
        nodes[38   ] = std::make_pair<double>(0.287362487355455553661443, 0.06203942315989266487186171);
        nodes[39   ] = std::make_pair<double>(-0.287362487355455553661443, 0.06203942315989266487186171);
        nodes[40   ] = std::make_pair<double>(0.2247637903946890503004141, 0.06311419228625402000343314);
        nodes[41   ] = std::make_pair<double>(-0.2247637903946890503004141, 0.06311419228625402000343314);
        nodes[42   ] = std::make_pair<double>(0.1612223560688917090022443, 0.06392423858464818531288643);
        nodes[43   ] = std::make_pair<double>(-0.1612223560688917090022443, 0.06392423858464818531288643);
        nodes[44   ] = std::make_pair<double>(0.09700469920946269697381581, 0.0644661644359500879408742);
        nodes[45   ] = std::make_pair<double>(-0.09700469920946269697381581, 0.0644661644359500879408742);
        nodes[46   ] = std::make_pair<double>(0.03238017096286936041815707, 0.06473769681268391751327584);
        nodes[47   ] = std::make_pair<double>(-0.03238017096286936041815707, 0.06473769681268391751327584);

        return nodes;
    }


    throw std::runtime_error("Invalid num_nodes passed to gauss_legendre_nodes");
}


inline
void leggauss(const int deg, std::vector<double> &x_smpl_org, std::vector<double> &weight_org) {
  std::vector<std::pair<double, double> > nodes = gauss_legendre_nodes(deg);

  x_smpl_org.resize(deg);
  weight_org.resize(deg);
  for (int n = 0; n < deg; ++n) {
    x_smpl_org[n] = nodes[n].first;
    weight_org[n] = nodes[n].second;
  }
}

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

inline
std::size_t find_section(const multi_array<double, 1> &section_edges, double x) {
  std::size_t idx = std::upper_bound(
      section_edges.origin(),
      section_edges.origin() + section_edges.num_elements(),
      x) - section_edges.origin() - 1;

  return std::min(idx, section_edges.num_elements() - 2);
}

inline
double interpolate_impl(double dx, const multi_array<double, 1> &coeffs) {
  double value = 0.0;
  double dx_power = 1.0;
  std::size_t N = coeffs.num_elements();
  for (int p = 0; p < N; ++p) {
    value += dx_power * coeffs(p);
    dx_power *= dx;
  }
  return value;
}

inline
void interpolate_all_l_impl(double dx, const multi_array<double, 2> &coeffs, std::vector<double> &result) {
  assert(result.size() == coeffs.extent(1));
  std::fill(result.begin(), result.end(), 0.0);
  double dx_power = 1.0;
  std::size_t N = coeffs.extent(0);
  std::size_t dim = coeffs.extent(1);
  for (int p = 0; p < N; ++p) {
    for (int l = 0; l < dim; ++l) {
      result[l] += dx_power * coeffs(p, l);
    }
    dx_power *= dx;
  }
}

inline
double interpolate(double x, const multi_array<double, 2> &_data, const multi_array<double, 1> &section_edges) {
  std::size_t section_idx = find_section(section_edges, x);
  return interpolate_impl(x - section_edges(section_idx), _data.make_view(section_idx));
};

inline
void interpolate_all_l(double x, const multi_array<double, 3> &_data,
                       const multi_array<double, 1> &section_edges, std::vector<double> &result) {
  std::size_t section_idx = find_section(section_edges, x);
  interpolate_all_l_impl(x - section_edges(section_idx), _data.make_view(section_idx), result);
};

inline
multi_array<double, 1>
differentiate_coeff(const multi_array<double, 1> &coeffs, std::size_t order) {
  std::size_t k = coeffs.num_elements();
  multi_array<double, 1> coeffs_deriv(coeffs);//this always makes a copy
  assert(coeffs_deriv.num_elements() == k);
  for (int o = 0; o < order; ++o) {
    for (int p = 0; p < k - 1 - o; ++p) {
      coeffs_deriv(p) = (p + 1) * coeffs_deriv(p + 1);
    }
    coeffs_deriv(k - 1 - o) = 0;
  }
  return coeffs_deriv;
}

inline
double interpolate_derivative(double x,
                              std::size_t order,
                              const multi_array<double, 2> &_data,
                              const multi_array<double, 1> &section_edges,
                              int section = -1) {
  using namespace internal;
  std::size_t section_idx = section >= 0 ? section : find_section(section_edges, x);
  multi_array<double, 1> coeffs = differentiate_coeff(_data.make_view(section_idx), order);
  return interpolate_impl(x - section_edges(section_idx), coeffs);
}

inline
void interpolate_derivative_all(double x,
                                const multi_array<double, 2> &_data,
                                const multi_array<double, 1> &section_edges,
                                std::vector<double> &result,
                                int section = -1) {
  using namespace internal;
  std::size_t section_idx = section >= 0 ? section : find_section(section_edges, x);
  multi_array<double, 1> coeffs = _data.make_view(section_idx);
  int k = coeffs.extent(0);
  multi_array<double, 1> coeffs_deriv(coeffs); // make copy for coeffs
  result.resize(k);
  for (int o = 0; o < k; o++) {
    result[o] = interpolate_impl(x - section_edges(section_idx), coeffs_deriv);
    for (int p = 0; p < k - 1; p++) {
      coeffs_deriv(p) = (p + 1) * coeffs_deriv(p + 1);
    }
    coeffs_deriv(k - 1) = 0;
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

inline
void
compute_unl_high_freq(const std::vector<bool> &mask,
                      const std::vector<double> &w_vec_org,
                      const multi_array<double, 2> &derive0,
                      const multi_array<double, 2> &derive1,
                      const double x0,
                      const double x1,
                      multi_array<std::complex<double>, 2> &result) {
  std::vector<double> w_vec_target;
  std::vector<int> idx_w_target;
  for (int i = 0; i < w_vec_org.size(); i++) {
    if (mask[i]) {
      w_vec_target.push_back(w_vec_org[i]);
      idx_w_target.push_back(i);
    }
  }
  std::size_t nw_target = w_vec_target.size();
  std::size_t nl = derive0.extent(0);
  std::size_t n_deriv = derive0.extent(1);
  std::complex<double> J(0.0, 1.0);

  assert(result.extent(0) == w_vec_org.size());
  assert(result.extent(1) == nl);

  //auto t1 = std::chrono::system_clock::now();

  std::vector<std::complex<double> > inv_iw(nw_target);
  std::vector<std::complex<double> > exp_w_x10(nw_target);
  std::vector<std::complex<double> > exp_w_x1(nw_target);
  std::vector<std::complex<double> > exp_w_x0(nw_target);
  for (int w = 0; w < nw_target; w++) {
    inv_iw[w] = 1.0/(J * w_vec_target[w]);
    exp_w_x10[w] = exp(J * w_vec_target[w] * (x1 - x0));
    exp_w_x0[w] = exp(J * w_vec_target[w] * x0);
    exp_w_x1[w] = exp(J * w_vec_target[w] * x1);
  }

  multi_array<std::complex<double>, 3> coeff(n_deriv, nw_target, nl);
  for (int k = 0; k < n_deriv; ++k) {
     for (int w = 0; w < nw_target; w++) {
       for (int l = 0; l < nl; l++) {
          coeff(k, w, l) = exp_w_x1[w] * derive1(l, k) - exp_w_x0[w] * derive0(l, k);
      }
    }
  }

  //auto t2 = std::chrono::system_clock::now();

  multi_array<std::complex<double>, 2> jk(nw_target, nl);
  jk.fill(0);
  for (int k = n_deriv - 1; k > -1; --k) {
    multi_array<std::complex<double>, 2> coeff_view = coeff.make_view(k);
    for (int w = 0; w < nw_target; w++) {
      for (int l = 0; l < nl; l++) {
        jk(w, l) = (coeff_view(w, l) - jk(w, l)) * inv_iw[w];
      }
    }
  }

  //auto t3 = std::chrono::system_clock::now();
  for (int i = 0; i < nw_target; i++) {
    for (int l = 0; l < nl; l++) {
      // use += to sum up contributions from all sections
      result(idx_w_target[i], l) += jk(i, l);
    }
  }
  //std::cout << "t2-t1 " <<  std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << std::endl;
  //std::cout << "t3-t2 " <<  std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << std::endl;
  //std::cout << "t4-t3 " <<  std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << std::endl;
}

struct func {
  void load_from_h5(hid_t file, const std::string &prefix) {
    data = internal::load_multi_array<double, 3>(file, prefix + std::string("/data"));
    np = internal::hdf5_read_scalar<int>(file, prefix + std::string("/np"));
    ns = internal::hdf5_read_scalar<int>(file, prefix + std::string("/ns"));
    nl = data.extent(0);
    section_edges = internal::load_multi_array<double, 1>(file, prefix + std::string("/section_edges"));

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
  multi_array<double, 1> section_edges;
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
  }

  /**
    * Return number of basis functions
    * @return  number of basis functions
    */
  int dim() const { return dim_; }

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
      return interpolate(x, ulx_.data.make_view(l), ulx_.section_edges);
    } else {
      return interpolate(-x, ulx_.data.make_view(l), ulx_.section_edges) * even_odd_sign(l);
    }
  }

  void ulx_all_l(double x, std::vector<double> &ulx_result) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(x) > 1) {
        throw std::runtime_error("x must be in [-1,1]!");
    }

    ulx_result.resize(ulx_.nl);
    interpolate_all_l(std::abs(x), ulx_.data_for_vec, ulx_.section_edges, ulx_result);
    if (x < 0) {
      for (int l = 1; l < ulx_.nl; l += 2) {
        ulx_result[l] *= -1;
      }
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
      return interpolate_derivative(x, order, ulx_.data.make_view(l), ulx_.section_edges);
    } else {
      return interpolate_derivative(-x, order, ulx_.data.make_view(l), ulx_.section_edges) * even_odd_sign(l + order);
    }
  }

  void d_ulx_all(int l, double x, std::vector<double> &d_ulx_result, const int section = -1) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(x) > 1) {
        throw std::runtime_error("x must be in [-1,1]!");
    }
    assert(x >= 0);
    interpolate_derivative_all(x, ulx_.data.make_view(l), ulx_.section_edges, d_ulx_result, section);
  }

  double vly(int l, double y) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(y) > 1) {
        throw std::runtime_error("y must be in [-1,1]!");
    }
    if (y >= 0) {
      return interpolate(y, vly_.data.make_view(l), vly_.section_edges);
    } else {
      return interpolate(-y, vly_.data.make_view(l), vly_.section_edges) * even_odd_sign(l);
    }
  }

  double d_vly(int l, double y, std::size_t order) const throw(std::runtime_error) {
    using namespace internal;

    if (std::abs(y) > 1) {
        throw std::runtime_error("y must be in [-1,1]!");
    }
    if (y >= 0) {
      return interpolate_derivative(y, order, vly_.data.make_view(l), vly_.section_edges);
    } else {
      return interpolate_derivative(-y, order, vly_.data.make_view(l), vly_.section_edges) * even_odd_sign(l + order);
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
    return ulx_.section_edges(index);
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

    int num_n = n.size();
    std::complex<double> J = std::complex<double>(0.0, 1.0);

    std::vector<double> o_vec(n.size());
    if (this->statistics_ == "F") {
      for (int i = 0; i < num_n; i++) {
        o_vec[i] = (2.0 * n[i] + 1.0);
      }
    } else {
      for (int i = 0; i < num_n; i++) {
        o_vec[i] = (2.0 * n[i]);
      }
    }

    //w_vec = 0.5 * M_PI * o_vec
    std::vector<double> w_vec(o_vec);
    std::transform(w_vec.begin(), w_vec.end(), w_vec.begin(), std::bind1st(std::multiplies<double>(), 0.5 * M_PI));

    std::size_t num_deriv = this->ulx_.data.extent(2);
    //Compute tail
    std::vector<std::vector<int> > replaced_with_tail(num_n, std::vector<int>(this->dim_, 0));
    multi_array<double, 2> deriv_x1(this->dim_, num_deriv);
    deriv_x1.fill(0.0);
    std::vector<double> d_ulx_result;
    for (int l = 0; l < dim_; ++l) {
      d_ulx_all(l, 1.0, d_ulx_result);
      for (int p = 0; p < num_deriv; ++p) {
        deriv_x1(l, p) = d_ulx_result[p];
      }
    }
    multi_array<std::complex<double>, 2> unl_tail = compute_unl_tail(w_vec, statistics_, deriv_x1, -1);
    multi_array<std::complex<double>, 2> unl_tail_without_last_two = compute_unl_tail(w_vec, statistics_, deriv_x1, 2);

    for (int i = 0; i < num_n; i++) {
      if (statistics_ == "B" && n[i] == 0)
        continue;
      for (int l = 0; l < dim_; ++l) {
        if (std::abs((unl_tail(i, l) - unl_tail_without_last_two(i, l)) / unl_tail(i, l)) < 1e-10) {
          replaced_with_tail[i][l] = 1;
        }
      }
    }

    std::vector<bool> mask_tail(num_n);
    for (int i = 0; i < num_n; ++i) {
      int tmp = 1;
      for (int l = 0; l < dim_; ++l) {
        tmp *= replaced_with_tail[i][l];
      }
      mask_tail[i] = (tmp == 0);
    }

    // Numerical integration
    multi_array<std::complex<double>, 2> result(num_n, dim_);
    result.fill(0);

    multi_array<double, 2> deriv0(dim_, num_deriv);
    multi_array<double, 2> deriv1(dim_, num_deriv);
    int deg = 24;
    std::vector<double> x_smpl_org;
    std::vector<double> weight_org;
    leggauss(deg, x_smpl_org, weight_org);
    std::vector<double> x_smpl(deg, 0.0);
    std::vector<double> weight(deg, 0.0);
    for (int s = 0; s < num_sections_x(); s++) {
      double x0 = ulx_.section_edges(s);
      double x1 = ulx_.section_edges(s + 1);
      //Derivatives at end points
      std::vector<double> _d_ulx_all_x0;
      std::vector<double> _d_ulx_all_x1;
      for (int l = 0; l < dim_; l++) {
        d_ulx_all(l, x0, _d_ulx_all_x0, s);
        d_ulx_all(l, x1, _d_ulx_all_x1, s);
        for (int p = 0; p < num_deriv; p++) {
          deriv0(l, p) = _d_ulx_all_x0[p];
          deriv1(l, p) = _d_ulx_all_x1[p];
        }
      }

      //Mask based on phase shift
      std::vector<bool> mask_use_high_freq_formula(num_n);
      for (int i = 0; i < num_n; i++) {
        mask_use_high_freq_formula[i] = (mask_tail[i] && ((std::abs(w_vec[i]) * (x1 - x0)) > (0.1 * M_PI)));
      }
      //High frequency formula
      compute_unl_high_freq(mask_use_high_freq_formula, w_vec, deriv0, deriv1, x0, x1, result);

      //low frequency formula (Gauss-Legendre quadrature)
      std::vector<bool> mask_use_low_freq_formula(num_n);
      std::vector<int> n_low_freq_formula;
      for (int i = 0; i < num_n; i++) {
        mask_use_low_freq_formula[i] = static_cast<bool>((!mask_use_high_freq_formula[i]) && mask_tail[i]);
        if (mask_use_low_freq_formula[i]) {
          n_low_freq_formula.push_back(i);
        }
      }

      int num_n_low_freq_formula = std::count(mask_use_low_freq_formula.begin(), mask_use_low_freq_formula.end(), true);
      assert(num_n_low_freq_formula == n_low_freq_formula.size());
      if (num_n_low_freq_formula > 0) {
        for (int i = 0; i < deg; i++) {
          x_smpl[i] = 0.5 * (x_smpl_org[i] + 1) * (x1 - x0) + x0;
          weight[i] = weight_org[i] * (x1 - x0) / 2.0;
        }
        multi_array<dcomplex, 2> smp_vals(deg, dim_);
        std::vector<double> ulx_result(dim_);
        smp_vals.fill(0.0);
        for (int ix = 0; ix < deg; ix++) {
          ulx_all_l(x_smpl[ix], ulx_result);
          for (int l = 0; l < dim_; l++) {
            smp_vals(ix, l) = ulx_result[l];
          }
        }

        multi_array<dcomplex, 2> weight_exp_iwx(num_n_low_freq_formula, deg);
        for (int i = 0; i < num_n_low_freq_formula; ++i) {
          for (int ix = 0; ix < deg; ix++) {
            weight_exp_iwx(i, ix) = weight[ix] * std::exp(J * w_vec[n_low_freq_formula[i]] * x_smpl[ix]);
          }
        }
        multi_array<dcomplex, 2> result_low_freq;
        multiply(weight_exp_iwx, smp_vals, result_low_freq);

        for (int i = 0; i < num_n_low_freq_formula; ++i) {
          for (int l = 0; l < dim_; l++) {
            // use += to sum up contributions from all sections
            result(n_low_freq_formula[i], l) += result_low_freq(i, l);
          }
        }

      }

    }//loop over section

    for (int l = 0; l < dim_; l++) {
      if (l % 2 == 0) {
        for (int i = 0; i < num_n; i++) {
          result(i, l) = result(i, l).real();
        }
      } else {
        for (int i = 0; i < num_n; i++) {
          result(i, l) = J * result(i, l).imag();
        }
      }
    }

    std::vector<std::vector<std::complex<double> > > result_vec(num_n, std::vector<std::complex<double> >(dim_, 0));
    for (int i = 0; i < num_n; i++) {
      for (int l = 0; l < dim_; l++) {
        result_vec[i][l] = sqrt(2.0) * exp(J * w_vec[i]) * result(i, l);
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
    return vly_.section_edges(index);
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


