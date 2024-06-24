// Copyright 2024 Eloi Dieme
#ifndef SRC_VALUE_HPP_
#define SRC_VALUE_HPP_

#include <functional>
#include <ostream>
#include <vector>

namespace dgrad {
class Value {
 public:
  explicit Value(double value) : _backward([]() {}), val(value), grad(0.0) {}

  friend std::ostream& operator<<(std::ostream& os, const Value& self) {
    os << "Value(data=" << self.val << ",grad=" << self.grad << ")";
    return os;
  }
  void backward();

  std::function<void()> _backward;
  std::vector<Value*> children;
  double val;
  double grad;
};

// Operations
Value add(Value* lhs, Value* rhs);
Value mult(Value* lhs, Value* rhs);
Value exp(Value* in);
Value tanh(Value* in);
Value relu(Value* in);
}  // namespace dgrad

#endif  // SRC_VALUE_HPP_
