// Copyright 2024 Eloi Dieme
#include <iostream>

#include "Value.hpp"

int main(int argc, char *argv[argc + 1]) {
  Value myVal = 4.2;
  Value myVal2 = 3.6;
  Value myVal3 = 1.2;

  Value result1 = myVal * myVal3;
  Value result = result1 + myVal2;
  result.backward();

  std::cout << result;
  std::cout << myVal;
  std::cout << myVal2;
  std::cout << myVal3;

  return EXIT_SUCCESS;
}
