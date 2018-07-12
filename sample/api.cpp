#include <iostream>
#include <vector>
#include "irbasis.hpp"

int main() {
    double Lamb = 1000.0;
    double x = 1.0;
    double y = 1.0;
    int l = 0;
    irbasis::basis b = irbasis::load("F", Lamb, "./irbasis.h5");
    
    //Dimensions of basis
    std::cout << b.dim() << std::endl;
    
    //u_l(x = 1) and v_l(y = 1)
    std::cout << b.ulx(l,x) << std::endl;
    std::cout << b.vly(l,y) << std::endl;
    
    //Singular value s_0
    std::cout << b.sl(l) << std::endl;
    
    //The k-th derivative of ulx and vly
    for (int k=1; k < 4; ++k) {
        std::cout << b.d_ulx(l,k,x) << std::endl;
        std::cout << b.d_vly(l,k,y) << std::endl;
    }
    
    //Fourier taransform of ulx
    std::vector<long long> vec;
    for (int n=0; n<1000; ++n) {
        vec.push_back(n);
    }

    //unl will be a two-dimensional array of size (vec.size(), b.dim)).
    std::vector<std::vector<std::complex<double> > >  unl = b.compute_unl(vec);
    std::cout << "Dimensions of unl " << unl.size() << " " << unl[0].size() << std::endl;
}
