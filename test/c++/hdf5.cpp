#include <fstream>

#include "gtest.h"
#include "../../c++/irbasis.hpp"

TEST(hdf5, read_double) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    double data = internal::hdf5_read_double(file, std::string("/test_data/double"));
    ASSERT_EQ(data, 100.0);
    H5Fclose(file);
}

TEST(hdf5, read_int) {
    std::string file_name("hdf5_test.h5");
    hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    int data = internal::hdf5_read_int(file, std::string("/test_data/int"));
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
    for (int i=0; i < data.size(); ++i) {
        ASSERT_EQ(data[i], i);
    }
    ASSERT_EQ(extents[0] * extents[1] * extents[2], data.size());
    H5Fclose(file);
}

/*
template<int DIM>
struct int_helper {};

template<>
int_helper<1> {
    static int 
}

template<class T>
class TypedTest : public testing::Test {};

typedef ::testing::Types<1, 2, 3> TestTypes;

TYPED_TEST_CASE(TypedTest , TestTypes);

TYPED_TEST(TypedTest, Test1)
{
        //TypeParam p = 0;
}
*/
