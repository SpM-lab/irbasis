#include <iostream>
#include <vector>
#include "irbasis.hpp"

int main() {
    double Lamb = 1000.0;
    irbasis::basis b = irbasis::load("F", Lamb, "./irbasis.h5");
    
    //Dimensions of basis
    std::cout << b.dim() << std::endl;
    
    //u_l(x = 1) and v_l(y = 1)
    std::cout << b.ulx(0,1) << std::endl;
    std::cout << b.vly(0,1) << std::endl;
    
    //Singular value s_0
    std::cout << b.sl(0) << std::endl;
    
    //The k-th derivative of ulx and vly
    std::cout << b.d_ulx(0,1,1) << std::endl;
    std::cout << b.d_vly(0,1,1) << std::endl;
    
    //Fourier taransform of ulx
    std::vector<long long> vec;
    for (int n=0; n<10000; ++n) {
        vec.push_back(n);
    }

    std::cout << b.compute_Tnl(vec)[0][0] << std::endl;

    std::cout << b.compute_Tnl(0)[0][0] << std::endl;
    
}
