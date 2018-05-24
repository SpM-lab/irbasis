#include <fstream>

#include "gtest.h"

#include "../../c++/irbasis.hpp"

using namespace irbasis;

TEST(interpolation, check_ulx_b) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  std::vector <std::vector <double> > ref_data10 = b10.check_ulx();
  for (int i=0; i<ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  std::vector <std::vector <double> > ref_data10000 = b10000.check_ulx();
  for (int i=0; i<ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, ulx_all_l) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  int Nl = b10.dim();
  std::vector<double> ul(Nl);
  double x = -0.1;
  b10.ulx_all_l(x, ul);
  for (int l=0; l<Nl; ++l) {
      ASSERT_NEAR(ul[l], b10.ulx(l, x), 1e-10);
  }
}

TEST(interpolation, check_ulx_f) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  std::vector <std::vector <double> > ref_data10 = b10.check_ulx();
  for (int i=0; i<ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  std::vector <std::vector <double> > ref_data10000 = b10000.check_ulx();
  for (int i=0; i<ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, check_vly_b) {
  basis b10("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
  std::vector <std::vector <double> > ref_data10 = b10.check_vly();
  for (int i=0; i<ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_b-mp-Lambda10000.0_np8");
  std::vector <std::vector <double> > ref_data10000 = b10000.check_vly();
  for (int i=0; i<ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, check_vly_f) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  std::vector <std::vector <double> > ref_data10 = b10.check_vly();
  for (int i=0; i<ref_data10.size(); i++) {
    ASSERT_LE(ref_data10[i][2], 1e-8);
  }
  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  std::vector <std::vector <double> > ref_data10000 = b10000.check_vly();
  for (int i=0; i<ref_data10000.size(); i++) {
    ASSERT_LE(ref_data10000[i][2], 1e-8);
  }
}

TEST(interpolation, differential_ulx) {
  basis b10("../irbasis.h5", "/basis_f-mp-Lambda10.0_np8");
  double d_1st_ref_data10 = b10.get_ref_ulx(1);
  double d_2nd_ref_data10 = b10.get_ref_ulx(2);
  int Nl = b10.dim();
  if (Nl % 2 == 1) Nl -= 1;
  ASSERT_LE(fabs((d_1st_ref_data10 - b10.d_ulx(Nl - 1, 1.0, 1)) / d_1st_ref_data10), 1e-8);
  ASSERT_LE(fabs((d_2nd_ref_data10 - b10.d_ulx(Nl - 1, 1.0, 2)) / d_2nd_ref_data10), 1e-8);

  basis b10000("../irbasis.h5", "/basis_f-mp-Lambda10000.0_np8");
  double d_1st_ref_data10000 = b10000.get_ref_ulx(1);
  double d_2nd_ref_data10000 = b10000.get_ref_ulx(2);
  Nl = b10000.dim();
  if (Nl % 2 == 1) Nl -= 1;
  ASSERT_LE(fabs((d_1st_ref_data10000 - b10000.d_ulx(Nl - 1, 1.0, 1)) / d_1st_ref_data10000), 1e-8);
  ASSERT_LE(fabs((d_2nd_ref_data10000 - b10000.d_ulx(Nl - 1, 1.0, 2)) / d_2nd_ref_data10000), 1e-8);
}
