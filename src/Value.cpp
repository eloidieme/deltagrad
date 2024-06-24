// Copyright 2024 Eloi Dieme
#include "Value.hpp"

#include <algorithm>
#include <functional>
#include <set>

#include "utils.hpp"

Value::Value(double value) {
  val = value;
  grad = 0.0;
  op = ' ';
  _backward = []() {};
}

Value Value::operator+(Value& other) {
  Value result = Value(val + other.val);
  result.children.push_back(this);
  result.children.push_back(&other);
  result.op = '+';

  double* res_grad = &result.grad;

  result._backward = [=, &other]() {
    this->grad += 1.0 * (*res_grad);
    other.grad += 1.0 * (*res_grad);
  };
  return result;
}

Value Value::operator*(Value& other) {
  Value result = Value(val * other.val);
  result.children.push_back(this);
  result.children.push_back(&other);
  result.op = '*';

  double* res_grad = &result.grad;

  result._backward = [=, &other]() {
    this->grad += other.val * (*res_grad);
    other.grad += this->val * (*res_grad);
  };

  return result;
}

void Value::backward() {
  std::vector<Value> topo;
  std::set<double> visited;

  build_topo(*this, topo, visited);
  grad = 1.0;
  std::reverse(topo.begin(), topo.end());
  for (Value v : topo) {
    v._backward();
  }
}
