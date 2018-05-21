#include <fstream>

#include "gtest.h"
#include "../../c++/irbasis.hpp"

TEST(hdf5, read_double) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    double data = internal::hdf5_read_scalar<double>(file, std::string("/test_data/double"));
    ASSERT_EQ(data, 100.0);
    H5Fclose(file);
}

TEST(hdf5, read_int) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    int data = internal::hdf5_read_scalar<int>(file, std::string("/test_data/int"));
    ASSERT_EQ(data, 100);
    H5Fclose(file);
}

TEST(hdf5, read_double_array1) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::vector<double> data;
    std::vector<std::size_t> extents;
    internal::hdf5_read_double_array<1>(file, std::string("/test_data/double_array1"), extents, data);
    for (int i=0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], i);
    }
    ASSERT_EQ(extents[0], data.size());
    H5Fclose(file);
}

TEST(hdf5, read_double_array2) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::vector<double> data;
    std::vector<std::size_t> extents;
    internal::hdf5_read_double_array<2>(file, std::string("/test_data/double_array2"), extents, data);
    for (int i=0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], i);
    }
    ASSERT_EQ(extents[0] * extents[1], data.size());
    H5Fclose(file);
}

TEST(hdf5, read_double_array3) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    std::vector<double> data;
    std::vector<std::size_t> extents;
    internal::hdf5_read_double_array<3>(file, std::string("/test_data/double_array3"), extents, data);
    internal::multi_array<double,3> a = internal::load_multi_array<double,3>(file, std::string("/test_data/double_array3"));
    for (int i=0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], i);
        ASSERT_EQ(*(a.origin()+i), data[i]);
    }
    ASSERT_EQ(extents[0] * extents[1] * extents[2], data.size());
    H5Fclose(file);
}

TEST(multi_array, dim2) {
    int N1 = 2;
    int N2 = 4;
    internal::multi_array<double,2> array(N1, N2);
    array(0, 0) = 0;
    array(N1-1, N2-1) = 0;
    ASSERT_EQ(array.extent(0), N1);
    ASSERT_EQ(array.extent(1), N2);
}

TEST(multi_array, array2_view) {
    int N1 = 2;
    int N2 = 4;
    internal::multi_array<double,2> array(N1, N2);

    for (int i=0; i < N1; ++i) {
        for (int j=0; j<N2; ++j) {
            array(i, j) = N2 * i + j;
        }
    }

    internal::multi_array<double,1> view = array.make_view(1);
    for (int j=0; j<N2; ++j) {
        ASSERT_EQ(view(j), N2+j);
    }
}

TEST(multi_array, array3_view) {
    int N1 = 2;
    int N2 = 4;
    int N3 = 8;
    internal::multi_array<double,3> array(N1, N2, N3);

    for (int i=0; i < N1; ++i) {
        for (int j=0; j<N2; ++j) {
            for (int k=0; k<N3; ++k) {
                array(i, j, k) = (i*N2 + j)*N3 + k;
            }
        }
    }

    internal::multi_array<double,2> view = array.make_view(1);
    for (int j=0; j<N2; ++j) {
        for (int k=0; k<N3; ++k) {
            ASSERT_EQ(view(j, k), array(1,j,k));
        }
    }
}

TEST(multi_array, sl_out) {
  basis b("../irbasis.h5", "/basis_b-mp-Lambda10.0_np8");
    for (int i=0; i<5; i++) {
        std::cout << b.sl(i) << std::endl;
    }
    std::cout << b.ulx(0, 0.5) << std::endl;
    std::cout << b.ulx(0, -0.5) << std::endl;
}

TEST(multi_array, check_ulx_b) {
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

TEST(multi_array, check_ulx_f) {
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
