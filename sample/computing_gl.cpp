#include <iostream>

#include "irbasis.hpp"

int main() {
  irbasis::basis b = irbasis::load("F", 1000.0, "./irbasis.h5");
  std::cout << b.dim() << std::endl;
}
