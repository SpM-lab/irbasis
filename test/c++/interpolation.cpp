#include <fstream>

#include "gtest.h"

#include "../../c++/irbasis.hpp"

using namespace irbasis;

class refdata {
public:
  refdata() {}

  refdata(
      const std::string &file_name,
      const std::string &prefix = ""
  ) throw(std::runtime_error) {
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

    if (file < 0) {
      throw std::runtime_error("Failed to open " + file_name + "!");
    }

    //read info
    Lambda = internal::hdf5_read_scalar<double>(file, prefix + std::string("/info/Lambda"));
    dim = internal::hdf5_read_scalar<int>(file, prefix + std::string("/info/dim"));

    //read unl_odd
    internal::multi_array<std::complex<double>, 2>
        data_odd = internal::load_multi_array<std::complex<double>, 2>(file, prefix + std::string("/data/lodd/unl"));
    unl_odd.resize(data_odd.extent(0));
    n_odd.resize(data_odd.extent(0));
    for (int i = 0; i < data_odd.extent(0); i++) {
      n_odd[i] = static_cast<long long> (data_odd(i, 0).real());
      unl_odd[i] = data_odd(i, 1);
    }

    unl_odd_max = internal::hdf5_read_scalar<double>(file, prefix + std::string("/data/lodd/unlmax"));
    odd_l = internal::hdf5_read_scalar<int>(file, prefix + std::string("/data/lodd/l"));

    //read unl_even
    internal::multi_array<std::complex<double>, 2>
        data_even = internal::load_multi_array<std::complex<double>, 2>(file, prefix + std::string("/data/leven/unl"));
    unl_even_max = internal::hdf5_read_scalar<double>(file, prefix + std::string("/data/leven/unlmax"));
    even_l = internal::hdf5_read_scalar<int>(file, prefix + std::string("/data/leven/l"));

    unl_even.resize(data_even.extent(0));
    n_even.resize(data_even.extent(0));
    for (int i = 0; i < data_odd.extent(0); i++) {
      n_even[i] = static_cast<long long> (data_even(i, 0).real());
      unl_even[i] = data_even(i, 1);
    }
    H5Fclose(file);
  }

  double Lambda;
  int dim;

  std::vector<std::complex<double> > unl_odd;
  std::vector<long long> n_odd;
  double unl_odd_max;
  int odd_l;

  std::vector<std::complex<double> > unl_even;
  std::vector<long long> n_even;
  double unl_even_max;
  int even_l;
};

TEST(interpolation, attributes) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  ASSERT_EQ(10.0, b10.Lambda());
  ASSERT_EQ("B", b10.statistics());
}

TEST(interpolation, check_ulx_b) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  std::vector<std::vector<double> > ref_data10 = b10.check_ulx();
  for (int i = 0; i < ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  std::vector<std::vector<double> > ref_data10000 = b10000.check_ulx();
  for (int i = 0; i < ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, ulx_all_l) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  int Nl = b10.dim();
  std::vector<double> ul(Nl);
  double x = -0.1;
  b10.ulx_all_l(x, ul);
  for (int l = 0; l < Nl; ++l) {
    ASSERT_NEAR(ul[l], b10.ulx(l, x), 1e-10);
  }
}

TEST(interpolation, check_ulx_f) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  std::vector<std::vector<double> > ref_data10 = b10.check_ulx();
  for (int i = 0; i < ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  std::vector<std::vector<double> > ref_data10000 = b10000.check_ulx();
  for (int i = 0; i < ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, check_vly_b) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  std::vector<std::vector<double> > ref_data10 = b10.check_vly();
  for (int i = 0; i < ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  std::vector<std::vector<double> > ref_data10000 = b10000.check_vly();
  for (int i = 0; i < ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, check_vly_f) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  std::vector<std::vector<double> > ref_data10 = b10.check_vly();
  for (int i = 0; i < ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  std::vector<std::vector<double> > ref_data10000 = b10000.check_vly();
  for (int i = 0; i < ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, differential_ulx) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  double d_1st_ref_data10 = b10.get_ref_ulx(1);
  double d_2nd_ref_data10 = b10.get_ref_ulx(2);
  int Nl = b10.dim();
  if (Nl % 2 == 1)
    Nl -= 1;
  ASSERT_LE(fabs((d_1st_ref_data10 - b10.d_ulx(Nl - 1, 1.0, 1)) / d_1st_ref_data10), 1e-8);
  ASSERT_LE(fabs((d_2nd_ref_data10 - b10.d_ulx(Nl - 1, 1.0, 2)) / d_2nd_ref_data10), 1e-8);

  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  double d_1st_ref_data10000 = b10000.get_ref_ulx(1);
  double d_2nd_ref_data10000 = b10000.get_ref_ulx(2);
  Nl = b10000.dim();
  if (Nl % 2 == 1)
    Nl -= 1;
  ASSERT_LE(fabs((d_1st_ref_data10000 - b10000.d_ulx(Nl - 1, 1.0, 1)) / d_1st_ref_data10000), 1e-8);
  ASSERT_LE(fabs((d_2nd_ref_data10000 - b10000.d_ulx(Nl - 1, 1.0, 2)) / d_2nd_ref_data10000), 1e-8);
}

double check_data_tail(basis bs, refdata rb, std::string _statics) {
  //Check odd-l
  int l = rb.odd_l;
  std::vector<long long> n(1, 1e+8);
  std::vector<std::vector<std::complex<double> > > unl = bs.compute_unl(n);
  double unl_limit, unl_coeff;
  if (_statics == "f") {
    unl_limit = -(bs.d_ulx(l, 1, 1) + bs.d_ulx(l, -1, 1)) / (M_PI * M_PI * sqrt(2.0));
    unl_coeff = unl[0][l].real() * n[0] * n[0];
  } else {
    unl_limit = -(bs.ulx(l, 1) - bs.ulx(l, -1)) / (M_PI * sqrt(2.0));
    unl_coeff = unl[0][l].imag() * n[0];
  }
  double dunl_coeff = std::abs(unl_limit - unl_coeff);
  if (std::abs(unl_limit) > 1e-12)
    dunl_coeff /= std::abs(unl_limit);

  //Check even-l
  l = rb.even_l;
  if (_statics == "f") {
    unl_limit = (bs.ulx(l, 1) + bs.ulx(l, -1)) / (M_PI * sqrt(2.0));
    unl_coeff = unl[0][l].imag() * n[0];
  } else {
    unl_limit = (bs.d_ulx(l, 1, 1) - bs.d_ulx(l, -1, 1)) / (M_PI * M_PI * sqrt(2.0));
    unl_coeff = unl[0][l].real() * n[0] * n[0];
  }
  double dunl_coeff_even = std::abs(unl_limit - unl_coeff);
  if (std::abs(unl_limit) > 1e-12)
    dunl_coeff_even /= std::abs(unl_limit);
  if (dunl_coeff_even > dunl_coeff)
    dunl_coeff = dunl_coeff_even;
  return dunl_coeff;
}

double check_data(basis bs, refdata rb, std::string _statics) {
  //Check odd-l
  int l = rb.odd_l;
  std::vector<std::vector<std::complex<double> > > unl = bs.compute_unl(rb.n_odd);
  double dunl_max = std::abs(unl[0][l] - rb.unl_odd[0]);
  for (int i = 1; i < rb.unl_odd.size(); i++) {
    double tmp = std::abs(unl[i][l] - rb.unl_odd[i]);
    if (tmp > dunl_max)
      dunl_max = tmp;
  }
  dunl_max /= std::abs(rb.unl_odd_max);

  //Check even-l
  l = rb.even_l;
  unl = bs.compute_unl(rb.n_even);
  double dunl_max_even = std::abs(unl[0][l] - rb.unl_even[0]);
  for (int i = 1; i < rb.unl_even.size(); i++) {
    double tmp = std::abs(unl[i][l] - rb.unl_even[i]);
    if (tmp > dunl_max_even)
      dunl_max_even = tmp;
  }
  dunl_max_even /= std::abs(rb.unl_even_max);
  if (dunl_max < dunl_max_even)
    dunl_max = dunl_max_even;

  return dunl_max;
}

TEST(interpolation, unl_limit) {

  basis b10f("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  refdata ref10f("../unl_safe_ref.h5", "/basis_f-mp-Lambda10.0");
  double dunl_coeff = check_data_tail(b10f, ref10f, "f");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10000f("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  refdata ref10000f("../unl_safe_ref.h5", "/basis_f-mp-Lambda10000.0");
  dunl_coeff = check_data_tail(b10000f, ref10000f, "f");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10b("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  refdata ref10b("../unl_safe_ref.h5", "/basis_b-mp-Lambda10.0");
  dunl_coeff = check_data_tail(b10b, ref10b, "b");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10000b("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  refdata ref10000b("../unl_safe_ref.h5", "/basis_b-mp-Lambda10000.0");
  dunl_coeff = check_data_tail(b10000b, ref10000b, "b");
  ASSERT_LE(dunl_coeff, 1e-7);
}

TEST(interpolation, unl) {

  basis b10f("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  refdata ref10f("../unl_safe_ref.h5", "/basis_f-mp-Lambda10.0");
  double dunl_coeff = check_data(b10f, ref10f, "f");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10000f("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  refdata ref10000f("../unl_safe_ref.h5", "/basis_f-mp-Lambda10000.0");
  dunl_coeff = check_data(b10000f, ref10000f, "f");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10b("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  refdata ref10b("../unl_safe_ref.h5", "/basis_b-mp-Lambda10.0");
  dunl_coeff = check_data(b10b, ref10b, "b");
  ASSERT_LE(dunl_coeff, 1e-7);

  basis b10000b("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  refdata ref10000b("../unl_safe_ref.h5", "/basis_b-mp-Lambda10000.0");
  dunl_coeff = check_data(b10000b, ref10000b, "b");
  ASSERT_LE(dunl_coeff, 1e-7);
}
