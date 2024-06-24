// Copyright 2024 Eloi Dieme
#include <iostream>

#include "Value.hpp"

using dgrad::Value;

int main(int argc, char *argv[argc + 1]) {
  Value myVal = Value(4.2);
  Value myVal2 = Value(3.6);
  Value myVal3 = Value(1.2);

  Value result1 = dgrad::mult(&myVal, &myVal3);
  Value result2 = dgrad::add(&result1, &myVal2);
  Value result = dgrad::exp(&result2);
  result.backward();

  std::cout << result << '\n';
  std::cout << myVal << '\n';
  std::cout << myVal2 << '\n';
  std::cout << myVal3 << '\n';

  return EXIT_SUCCESS;
}
