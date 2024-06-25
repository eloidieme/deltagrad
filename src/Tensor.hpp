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
      : _zero_grad([=]() {
          for (size_t i = 0; i < size; ++i) {
            this->grad[i] = 0.0;
          }
        }),
        _backward([]() {}),
        val(value),
        grad(value.size(), 0.0),
        size(value.size()) {}

  explicit Tensor(std::vector<double> value)
      : _zero_grad([=]() {
          for (size_t i = 0; i < size; ++i) {
            this->grad[i] = 0.0;
          }
        }),
        _backward([]() {}),
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
    os << ",grad=[";
    for (size_t i = 0; i < self.size; ++i) {
      if (i == self.size - 1) {
        os << self.grad[i] << ']';
      } else {
        os << self.grad[i] << ',';
      }
    }
    os << ")";
    return os;
  }
  void backward();
  void zero_grad();
  static void set_nograd(const bool value) { nograd = value; }

  std::function<void()> _zero_grad;
  std::function<void()> _backward;
  std::vector<Tensor*> children;
  std::vector<double> val;
  std::vector<double> grad;
  double size;
  static bool nograd;
};

// Operations
Tensor add(Tensor* lhs, Tensor* rhs);
Tensor subs(Tensor* lhs, Tensor* rhs);
Tensor mult(Tensor* lhs, Tensor* rhs);
Tensor div(Tensor* lhs, Tensor* rhs);
Tensor dot(Tensor* lhs, Tensor* rhs);
Tensor exp(Tensor* in);
Tensor tanh(Tensor* in);
Tensor relu(Tensor* in);
}  // namespace dgrad

#endif  // SRC_TENSOR_HPP_
