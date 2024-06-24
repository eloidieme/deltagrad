#ifndef SRC_UTILS_HPP_
#define SRC_UTILS_HPP_

#include <set>
#include <vector>

#include "Value.hpp"

void build_topo(dgrad::Value v, std::vector<dgrad::Value> &topo,
                std::set<double> &visited) {
  if (visited.find(v.val) == visited.end()) {
    visited.insert(v.val);
    for (dgrad::Value *child : v.children) {
      build_topo(*child, topo, visited);
    }
    topo.push_back(v);
  }
}

#endif  // SRC_UTILS_HPP_
