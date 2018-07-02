#include <iostream>

#include "irbasis.hpp"

int main() {
	irbasis::basis b = irbasis::load("F", 1000.0, "./irbasis.h5");

	std::cout << b.dim() << std::endl;
	std::cout << b.ulx(0,1) << std::endl;
	std::cout << b.vly(0,1) << std::endl;
	std::cout << b.sl(0) << std::endl;
	std::cout << b.d_ulx(0,1,1) << std::endl;
	std::cout << b.d_vly(0,1,1) << std::endl;
}
