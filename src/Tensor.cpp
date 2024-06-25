// Copyright 2024 Eloi Dieme
#include "Tensor.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>

#include "utils.hpp"

dgrad::Tensor dgrad::add(dgrad::Tensor* lhs, dgrad::Tensor* rhs) {
  if (lhs->size != rhs->size) {
    std::cerr << "Sizes don't match for add operation\n";
    exit(-1);
  }
  std::vector<double> result_val = std::vector<double>(lhs->size, 0.0);
  for (size_t i = 0; i < lhs->size; ++i) {
    result_val[i] = lhs->val[i] + rhs->val[i];
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  std::vector<double>* res_grad = &result.grad;
  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < lhs->size; ++i) {
        lhs->grad[i] += 1.0 * (*res_grad)[i];
        rhs->grad[i] += 1.0 * (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::subs(dgrad::Tensor* lhs, dgrad::Tensor* rhs) {
  if (lhs->size != rhs->size) {
    std::cerr << "Sizes don't match for subs operation\n";
    exit(-1);
  }
  std::vector<double> result_val = std::vector<double>(lhs->size, 0.0);
  for (size_t i = 0; i < lhs->size; ++i) {
    result_val[i] = lhs->val[i] - rhs->val[i];
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < lhs->size; ++i) {
        lhs->grad[i] += 1.0 * (*res_grad)[i];
        rhs->grad[i] += (-1.0) * (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::mult(dgrad::Tensor* lhs, dgrad::Tensor* rhs) {
  if (lhs->size != rhs->size) {
    std::cerr << "Sizes don't match for mult operation\n";
    exit(-1);
  }
  std::vector<double> result_val = std::vector<double>(lhs->size, 0.0);
  for (size_t i = 0; i < lhs->size; ++i) {
    result_val[i] = lhs->val[i] * rhs->val[i];
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < lhs->size; ++i) {
        lhs->grad[i] += rhs->val[i] * (*res_grad)[i];
        rhs->grad[i] += lhs->val[i] * (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::div(dgrad::Tensor* lhs, dgrad::Tensor* rhs) {
  if (lhs->size != rhs->size) {
    std::cerr << "Sizes don't match for div operation\n";
    exit(-1);
  }
  std::vector<double> result_val = std::vector<double>(lhs->size, 0.0);
  for (size_t i = 0; i < lhs->size; ++i) {
    result_val[i] = lhs->val[i] / rhs->val[i];
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < lhs->size; ++i) {
        lhs->grad[i] += (1 / rhs->val[i]) * (*res_grad)[i];
        rhs->grad[i] +=
            -(lhs->val[i] / (rhs->val[i] * rhs->val[i])) * (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::dot(dgrad::Tensor* lhs, dgrad::Tensor* rhs) {
  if (lhs->size != rhs->size) {
    std::cerr << "Sizes don't match for dot operation\n";
    exit(-1);
  }
  std::vector<double> result_val = std::vector<double>(1, 0.0);
  for (size_t i = 0; i < lhs->size; ++i) {
    result_val[0] += lhs->val[i] * rhs->val[i];
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(lhs);
  result.children.push_back(rhs);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < lhs->size; ++i) {
        lhs->grad[i] += rhs->val[i] * (*res_grad)[0];
        rhs->grad[i] += lhs->val[i] * (*res_grad)[0];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::exp(dgrad::Tensor* in) {
  std::vector<double> result_val = std::vector<double>(in->size, 0.0);
  for (size_t i = 0; i < in->size; ++i) {
    result_val[i] = std::exp(in->val[i]);
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(in);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < in->size; ++i) {
        in->grad[i] += result.val[i] * (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::tanh(dgrad::Tensor* in) {
  std::vector<double> result_val = std::vector<double>(in->size, 0.0);
  for (size_t i = 0; i < in->size; ++i) {
    result_val[i] = std::tanh(in->val[i]);
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(in);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < in->size; ++i) {
        in->grad[i] +=
            (1 - std::tanh(result.val[i]) * std::tanh(result.val[i])) *
            (*res_grad)[i];
      }
    };
  }
  return result;
}

dgrad::Tensor dgrad::relu(dgrad::Tensor* in) {
  std::vector<double> result_val = std::vector<double>(in->size, 0.0);
  for (size_t i = 0; i < in->size; ++i) {
    result_val[i] = in->val[i] > 0 ? in->val[i] : 0.0;
  }
  dgrad::Tensor result = dgrad::Tensor(result_val);
  result.children.push_back(in);

  std::vector<double>* res_grad = &result.grad;

  if (!dgrad::Tensor::nograd) {
    result._backward = [=]() {
      for (size_t i = 0; i < in->size; ++i) {
        in->grad[i] += (in->val[i] > 0 ? 1 : 0) * (*res_grad)[i];
      }
    };
  }
  return result;
}

void dgrad::Tensor::backward() {
  std::vector<dgrad::Tensor> topo;
  std::set<std::vector<double>> visited;

  build_topo(*this, topo, visited);
  for (size_t i = 0; i < size; ++i) {
    grad[i] = 1.0;
  }
  std::reverse(topo.begin(), topo.end());
  for (dgrad::Tensor v : topo) {
    v._backward();
  }
}

void dgrad::Tensor::zero_grad() {
  std::vector<dgrad::Tensor> topo;
  std::set<std::vector<double>> visited;

  build_topo(*this, topo, visited);
  for (size_t i = 0; i < size; ++i) {
    grad[i] = 0.0;
  }
  std::reverse(topo.begin(), topo.end());
  for (dgrad::Tensor v : topo) {
    v._zero_grad();
  }
}
