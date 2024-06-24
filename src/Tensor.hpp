// Copyright 2024 Eloi Dieme
#ifndef SRC_TENSOR_HPP_
#define SRC_TENSOR_HPP_

#include <functional>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace dgrad {
class Tensor {
 public:
  explicit Tensor(std::initializer_list<double> value)
      : _backward([]() {}),
        val(value),
        grad(value.size(), 0.0),
        size(value.size()) {}

  explicit Tensor(std::vector<double> value)
      : _backward([]() {}),
        val(value),
        grad(value.size(), 0.0),
        size(value.size()) {}

  friend std::ostream& operator<<(std::ostream& os, const Tensor& self) {
    os << "Tensor(data=[";
    for (size_t i = 0; i < self.size; ++i) {
      if (i == self.size - 1) {
        os << self.val[i] << ']';
      } else {
        os << self.val[i] << ',';
      }
    }
    os << "],grad=[";
    for (size_t i = 0; i < self.size; ++i) {
      if (i == self.size - 1) {
        os << self.grad[i] << ']';
      } else {
        os << self.grad[i] << ',';
      }
    }
    os << "])";
    return os;
  }
  void backward();

  std::function<void()> _backward;
  std::vector<Tensor*> children;
  std::vector<double> val;
  std::vector<double> grad;
  double size;
};

// Operations
Tensor add(Tensor* lhs, Tensor* rhs);
Tensor mult(Tensor* lhs, Tensor* rhs);
Tensor exp(Tensor* in);
Tensor tanh(Tensor* in);
Tensor relu(Tensor* in);
}  // namespace dgrad

#endif  // SRC_TENSOR_HPP_
