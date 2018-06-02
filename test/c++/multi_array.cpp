#include <fstream>

#include "gtest.h"
#include "../../c++/irbasis.hpp"

using namespace irbasis;

TEST(multi_array, dim2) {
  int N1 = 2;
  int N2 = 4;
  internal::multi_array<double, 2> array(N1, N2);
  array(0, 0) = 0;
  array(N1 - 1, N2 - 1) = 0;
  ASSERT_EQ(array.extent(0), N1);
  ASSERT_EQ(array.extent(1), N2);
}

TEST(multi_array, array2_view) {
  int N1 = 2;
  int N2 = 4;
  internal::multi_array<double, 2> array(N1, N2);

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      array(i, j) = N2 * i + j;
    }
  }

  internal::multi_array<double, 1> view = array.make_view(1);
  for (int j = 0; j < N2; ++j) {
    ASSERT_EQ(view(j), N2 + j);
  }
}

TEST(multi_array, array3_view) {
  int N1 = 2;
  int N2 = 4;
  int N3 = 8;
  internal::multi_array<double, 3> array(N1, N2, N3);

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      for (int k = 0; k < N3; ++k) {
        array(i, j, k) = (i * N2 + j) * N3 + k;
      }
    }
  }

  internal::multi_array<double, 2> view = array.make_view(1);
  for (int j = 0; j < N2; ++j) {
    for (int k = 0; k < N3; ++k) {
      ASSERT_EQ(view(j, k), array(1, j, k));
    }
  }
}

TEST(multi_array, fill) {
  int N1 = 2;
  int N2 = 4;
  double value = 1.0;
  internal::multi_array<double, 2> array(N1, N2);
  array.fill(value);
  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      ASSERT_EQ(value, array(i, j));
    }
  }

}

TEST(multi_array, matrix_view) {
  int N1 = 2;
  int N2 = 4;
  int N3 = 8;
  internal::multi_array<double, 3> array(N1, N2, N3);

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      for (int k = 0; k < N3; ++k) {
        array(i, j, k) = (i * N2 + j) * N3 + k;
      }
    }
  }

  internal::multi_array<double, 2> view = array.make_matrix_view(N1 * N2, N3);

  int I = 0;
  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      for (int k = 0; k < N3; ++k) {
        ASSERT_EQ(view(I, k), (i * N2 + j) * N3 + k);
      }
      ++I;
    }
  }

}

TEST(multi_array, multiply) {
  int N1 = 2;
  int N2 = 3;
  int N3 = 3;
  internal::multi_array<double, 2> A(N1, N2);
  internal::multi_array<double, 2> B(N2, N3);
  internal::multi_array<double, 2> AB(N1, N3);
  internal::multi_array<double, 2> AB_test(N1, N3);

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      A(i, j) = i + 10 * j;
    }
  }

  for (int i = 0; i < N2; ++i) {
    for (int j = 0; j < N3; ++j) {
      B(i, j) = i + 9 * j;
    }
  }

  AB.fill(0);
  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      for (int k = 0; k < N3; ++k) {
        AB(i, k) += A(i, j) * B(j, k);
      }
    }
  }

  internal::multiply(A, B, AB_test);

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N3; ++j) {
      ASSERT_NEAR(AB(i, j), AB_test(i, j), 1e-10);
    }
  }

}
