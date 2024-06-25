// Copyright 2024 Eloi Dieme
#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <set>
#include <vector>

#include "Tensor.hpp"

void build_topo(dgrad::Tensor* v, std::vector<dgrad::Tensor*>& topo,
                std::set<dgrad::Tensor*>& visited) {
  if (visited.find(v) == visited.end()) {
    visited.insert(v);
    for (dgrad::Tensor* child : v->children) {
      build_topo(child, topo, visited);
    }
    topo.push_back(v);
  }
}

#endif  // SRC_UTILS_HPP_
