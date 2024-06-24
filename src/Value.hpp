// Copyright 2024 Eloi Dieme
#ifndef SRC_VALUE_HPP_
#define SRC_VALUE_HPP_

#include <functional>
#include <ostream>
#include <vector>

class Value {
 public:
  Value(double value);

  friend std::ostream& operator<<(std::ostream& os, const Value& self) {
    os << "Value(data=" << self.val << ",grad=" << self.grad << ")\n";
    return os;
  }
  Value operator+(Value& other);
  Value operator*(Value& other);
  void backward();

  std::function<void()> _backward;
  std::vector<Value*> children;
  double val;
  char op;
  double grad;
};

#endif  // SRC_VALUE_HPP_
