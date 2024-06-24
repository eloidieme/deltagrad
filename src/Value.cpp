// Copyright 2024 Eloi Dieme
#include "Value.hpp"

#include <algorithm>
#include <functional>
#include <set>

#include "utils.hpp"

dgrad::Value dgrad::add(dgrad::Value* lhs, dgrad::Value* rhs) {
  dgrad::Value result = dgrad::Value(lhs->val + rhs->val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  double* res_grad = &result.grad;

  result._backward = [=]() {
    lhs->grad += 1.0 * (*res_grad);
    rhs->grad += 1.0 * (*res_grad);
  };
  return result;
}

dgrad::Value dgrad::mult(dgrad::Value* lhs, dgrad::Value* rhs) {
  dgrad::Value result = dgrad::Value(lhs->val * rhs->val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  double* res_grad = &result.grad;

  result._backward = [=]() {
    lhs->grad += rhs->val * (*res_grad);
    rhs->grad += lhs->val * (*res_grad);
  };

  return result;
}

dgrad::Value dgrad::exp(dgrad::Value* in) {
  dgrad::Value result = dgrad::Value(std::exp(in->val));
  result.children.push_back(in);

  double* res_grad = &result.grad;

  result._backward = [=]() { in->grad += result.val * (*res_grad); };

  return result;
}

dgrad::Value dgrad::tanh(dgrad::Value* in) {
  dgrad::Value result = dgrad::Value(std::tanh(in->val));
  result.children.push_back(in);

  double* res_grad = &result.grad;

  result._backward = [=]() {
    in->grad +=
        (1 - std::tanh(result.val) * std::tanh(result.val)) * (*res_grad);
  };

  return result;
}

dgrad::Value dgrad::relu(dgrad::Value* in) {
  dgrad::Value result = dgrad::Value(in->val > 0 ? in->val : 0);
  result.children.push_back(in);

  double* res_grad = &result.grad;

  result._backward = [=]() { in->grad += (in->val > 0 ? 1 : 0) * (*res_grad); };

  return result;
}

void dgrad::Value::backward() {
  std::vector<dgrad::Value> topo;
  std::set<double> visited;

  build_topo(*this, topo, visited);
  grad = 1.0;
  std::reverse(topo.begin(), topo.end());
  for (dgrad::Value v : topo) {
    v._backward();
  }
}
