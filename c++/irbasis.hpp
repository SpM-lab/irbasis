#pragma once

#include <iostream>
#include <complex>
//#include <cmath>
#include <vector>
#include <set>
#include <assert.h>
#include <memory>
#include <fstream>

#include <hdf5.h>

namespace {

    namespace internal {
        // read double
        inline hdf5_read_double(const H5::H5File& file, const std::string& name, double& data) {
        }
    }

    class basis {
    public:
        /**
         * Constructor
         * @param v_basis piecewise polynomials representing v_l(y)
         */
        basis(
            const std::string& file_name,
            const std::string& prefix = "",
        ) throw(std::runtime_error) {
        }

    };

}
