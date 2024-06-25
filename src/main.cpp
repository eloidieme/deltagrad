// Copyright 2024 Eloi Dieme
#include <iostream>

#include "Tensor.hpp"

using dgrad::Tensor;

Tensor loss(Tensor x, Tensor y) {
  // mean squared error loss
  Tensor error = dgrad::subs(&x, &y);
  Tensor loss = dgrad::dot(&error, &error);
  Tensor n = Tensor({x.size});
  Tensor mse = dgrad::div(&loss, &n);
  return mse;
}

// forward declaration of static nograd flag
// necessary for tensor operations
bool Tensor::nograd;

int main(void) {
  /*
  Tensor X = Tensor({0, 1, 2, 3, 4, 5, 20, 100});
  Tensor Y = Tensor({0, 1, 4, 9, 16, 25, 400, 10000});
  Tensor W = Tensor({4.2, 9.1, 3.0, 1.7, 2.1, 3.5, 0.8, 1.3});
  Tensor b = Tensor({1.2});

  Tensor::set_nograd(false);
  Tensor prod = dgrad::dot(&W, &X);
  Tensor h = dgrad::add(&prod, &b);
  Tensor::set_nograd(true);
  Tensor v = dgrad::subs(&h, &b);
  Tensor::set_nograd(false);
  Tensor a = dgrad::tanh(&h);
  a.backward();
  a.zero_grad();

  prod = dgrad::dot(&W, &X);
  h = dgrad::add(&prod, &b);
  v = dgrad::subs(&h, &b);
  a = dgrad::tanh(&h);
  a.backward();

  std::cout << v << '\n';
  std::cout << a << '\n';
  std::cout << W << '\n';
  std::cout << X << '\n';
  std::cout << b << '\n';
  */

  Tensor a = Tensor({2.4});
  Tensor b = Tensor({1.2});
  Tensor c = dgrad::mult(&a, &b);
  Tensor d = dgrad::tanh(&c);
  d.backward();
  std::cout << a << '\n';
  std::cout << b << '\n';

  return EXIT_SUCCESS;
}
