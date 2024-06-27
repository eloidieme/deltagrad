// Copyright 2024 Eloi Dieme
#ifndef SRC_TENSOR_H_
#define SRC_TENSOR_H_

#define MAX_GRAPH_DEPTH 128

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

typedef enum Operation {
  UNKNOWN = -1,
  ADD,
  SUBS,
  MULT,
  DIV,
  DOT,
  EXP,
  TANH,
  RELU
} Operation;

typedef struct Tensor Tensor;
struct Tensor {
  uint32_t id;
  size_t size;
  Tensor* children[2];
  double* prev[2];
  double* grad;
  double* data;
  Operation op;
};

void dg_InitTensor(Tensor* tensor, size_t size);
void dg_PopulateTensor(Tensor* tensor, double* data, size_t dataLen);
void dg_DestroyTensor(Tensor* tensor);

void dg_backward(Tensor* tensor);
void dg_localDiff(Tensor* tensor);

void dg_repr(const char* label, Tensor* tensor);
Tensor dg_add(Tensor* lhs, Tensor* rhs);
Tensor dg_subs(Tensor* lhs, Tensor* rhs);
Tensor dg_mult(Tensor* lhs, Tensor* rhs);
Tensor dg_div(Tensor* lhs, Tensor* rhs);
Tensor dg_dot(Tensor* lhs, Tensor* rhs);
Tensor dg_exp(Tensor* in);
Tensor dg_tanh(Tensor* in);
Tensor dg_relu(Tensor* in);

bool dg_find(Tensor* target, Tensor* topo[], size_t size);
size_t build_topo(Tensor* v, Tensor* topo[], size_t topoSize, Tensor* visited[],
                  size_t visitedSize);

#endif  // SRC_TENSOR_H_
