// Copyright 2024 Eloi Dieme
#include <iostream>

#include "Tensor.hpp"

using dgrad::Tensor;

int main(int argc, char *argv[argc + 1]) {
  Tensor myVal = Tensor({4.2, 9.1, 3.0, 1.7});
  Tensor myVal2 = Tensor({3.6, 4.21, 0.8, 12});
  Tensor myVal3 = Tensor({1.2, 8.0, 9.1, 10.1});

  Tensor result1 = dgrad::mult(&myVal, &myVal3);
  Tensor result2 = dgrad::add(&result1, &myVal2);
  Tensor result = dgrad::tanh(&result2);
  result.backward();

  std::cout << result << '\n';
  std::cout << myVal << '\n';
  std::cout << myVal2 << '\n';
  std::cout << myVal3 << '\n';

  return EXIT_SUCCESS;
}
