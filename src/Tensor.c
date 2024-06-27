#include "Tensor.h"

static uint32_t lastAssigned = 0;

void dg_InitTensor(Tensor* tensor, size_t size) {
  tensor->id = ++lastAssigned;
  tensor->size = size;
  tensor->op = UNKNOWN;

  tensor->prev[0] = 0;
  tensor->prev[1] = 0;
  tensor->children[0] = NULL;
  tensor->children[1] = NULL;

  tensor->data = (double*)malloc(size * sizeof(double));
  tensor->grad = (double*)malloc(size * sizeof(double));

  for (size_t i = 0; i < size; ++i) {
    tensor->grad[i] = 0.0;
  }
}

void dg_PopulateTensor(Tensor* tensor, double* data, size_t dataLen) {
  if (dataLen != tensor->size) {
    perror("Size mismatch when populating tensor");
    exit(-1);
  }
  for (size_t i = 0; i < tensor->size; ++i) {
    tensor->data[i] = data[i];
  }
}
void dg_DestroyTensor(Tensor* tensor) {
  free(tensor->data);
  free(tensor->grad);
  free(tensor->prev[0]);
  free(tensor->prev[1]);
}

void dg_localDiff(Tensor* tensor) {
  switch (tensor->op) {
    case ADD:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] += 1.0 * tensor->grad[i];
        tensor->children[1]->grad[i] += 1.0 * tensor->grad[i];
      }
      break;
    case SUBS:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] += 1.0 * tensor->grad[i];
        tensor->children[1]->grad[i] += -1.0 * tensor->grad[i];
      }
      break;
    case MULT:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] += tensor->prev[1][i] * tensor->grad[i];
        tensor->children[1]->grad[i] += tensor->prev[0][i] * tensor->grad[i];
      }
      break;
    case DIV:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] +=
            (1 / tensor->prev[1][i]) * tensor->grad[i];
        tensor->children[1]->grad[i] +=
            -(tensor->prev[0][i] / (tensor->prev[1][i] * tensor->prev[1][i])) *
            tensor->grad[i];
      }
      break;
    case EXP:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] += tensor->data[i] * tensor->grad[i];
      }
      break;
    case DOT:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] += tensor->prev[1][i] * tensor->grad[0];
        tensor->children[1]->grad[i] += tensor->prev[0][i] * tensor->grad[0];
      }
      break;
    case TANH:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] +=
            (1 - tanh(tensor->data[i]) * tanh(tensor->data[i])) *
            tensor->grad[i];
      }
      break;
    case RELU:
      for (size_t i = 0; i < tensor->size; ++i) {
        tensor->children[0]->grad[i] +=
            (tensor->data[i] > 0 ? 1 : 0) * tensor->grad[i];
      }
      break;
    default:
      break;
  }
}

void dg_backward(Tensor* tensor) {
  Tensor* topo[MAX_GRAPH_DEPTH];
  Tensor* visited[MAX_GRAPH_DEPTH];

  size_t topoTop = build_topo(tensor, topo, 0, visited, 0);
  for (size_t i = 0; i < tensor->size; ++i) {
    tensor->grad[i] = 1.0;
  }

  for (signed i = topoTop - 1; i >= 0; --i) {
    dg_localDiff(topo[i]);
  }
}

void dg_repr(const char* label, Tensor* tensor) {
  printf("%s: Tensor(data=[", label);
  for (size_t i = 0; i < tensor->size - 1; ++i) {
    printf("%g,", tensor->data[i]);
  }
  printf("%g],grad=[", tensor->data[tensor->size - 1]);
  for (size_t i = 0; i < tensor->size - 1; ++i) {
    printf("%g,", tensor->grad[i]);
  }
  printf("%g])\n", tensor->grad[tensor->size - 1]);
}

bool dg_find(Tensor* target, Tensor* tarr[], size_t size) {
  if (size == 0) return false;
  bool found = false;
  for (size_t i = 0; i < size; ++i) {
    if (target->id == tarr[i]->id) {
      found = true;
      break;
    }
  }

  return found;
}
size_t build_topo(Tensor* v, Tensor* topo[], size_t topoTop, Tensor* visited[],
                  size_t visitedTop) {
  if (!dg_find(v, visited, visitedTop)) {
    visited[visitedTop] = v;
    for (size_t i = 0; i < 2; ++i) {
      if (v->children[i])
        topoTop =
            build_topo(v->children[i], topo, topoTop, visited, ++visitedTop);
    }
    topo[topoTop++] = v;
  }
  return topoTop;
}

Tensor dg_add(Tensor* lhs, Tensor* rhs) {
  if (lhs->size != rhs->size) {
    perror("Sizes don't match for add operation\n");
    exit(-1);
  }

  Tensor result;
  dg_InitTensor(&result, lhs->size);
  result.prev[0] = (double*)malloc(lhs->size * sizeof(double));
  result.prev[1] = (double*)malloc(lhs->size * sizeof(double));
  for (size_t i = 0; i < lhs->size; ++i) {
    result.data[i] = lhs->data[i] + rhs->data[i];
    result.prev[0][i] = lhs->data[i];
    result.prev[1][i] = rhs->data[i];
  }
  result.children[0] = lhs;
  result.children[1] = rhs;
  result.op = ADD;

  return result;
}

Tensor dg_subs(Tensor* lhs, Tensor* rhs) {
  if (lhs->size != rhs->size) {
    perror("Sizes don't match for subs operation\n");
    exit(-1);
  }

  Tensor result;
  dg_InitTensor(&result, lhs->size);
  result.prev[0] = (double*)malloc(lhs->size * sizeof(double));
  result.prev[1] = (double*)malloc(lhs->size * sizeof(double));
  for (size_t i = 0; i < lhs->size; ++i) {
    result.data[i] = lhs->data[i] - rhs->data[i];
    result.prev[0][i] = lhs->data[i];
    result.prev[1][i] = rhs->data[i];
  }
  result.children[0] = lhs;
  result.children[1] = rhs;
  result.op = SUBS;

  return result;
}
Tensor dg_mult(Tensor* lhs, Tensor* rhs) {
  if (lhs->size != rhs->size) {
    perror("Sizes don't match for mult operation\n");
    exit(-1);
  }

  Tensor result;
  dg_InitTensor(&result, lhs->size);
  result.prev[0] = (double*)malloc(lhs->size * sizeof(double));
  result.prev[1] = (double*)malloc(lhs->size * sizeof(double));
  for (size_t i = 0; i < lhs->size; ++i) {
    result.data[i] = lhs->data[i] * rhs->data[i];
    result.prev[0][i] = lhs->data[i];
    result.prev[1][i] = rhs->data[i];
  }
  result.children[0] = lhs;
  result.children[1] = rhs;
  result.op = MULT;

  return result;
}

Tensor dg_div(Tensor* lhs, Tensor* rhs) {
  if (lhs->size != rhs->size) {
    perror("Sizes don't match for div operation\n");
    exit(-1);
  }

  Tensor result;
  dg_InitTensor(&result, lhs->size);
  result.prev[0] = (double*)malloc(lhs->size * sizeof(double));
  result.prev[1] = (double*)malloc(lhs->size * sizeof(double));
  for (size_t i = 0; i < lhs->size; ++i) {
    result.data[i] = lhs->data[i] / rhs->data[i];
    result.prev[0][i] = lhs->data[i];
    result.prev[1][i] = rhs->data[i];
  }
  result.children[0] = lhs;
  result.children[1] = rhs;
  result.op = DIV;

  return result;
}

Tensor dg_dot(Tensor* lhs, Tensor* rhs) {
  if (lhs->size != rhs->size) {
    perror("Sizes don't match for dot operation\n");
    exit(-1);
  }

  Tensor result;
  dg_InitTensor(&result, 1);
  result.prev[0] = (double*)malloc(lhs->size * sizeof(double));
  result.prev[1] = (double*)malloc(lhs->size * sizeof(double));
  for (size_t i = 0; i < lhs->size; ++i) {
    result.data[0] += lhs->data[i] * rhs->data[i];
    result.prev[0][i] = lhs->data[i];
    result.prev[1][i] = rhs->data[i];
  }
  result.children[0] = lhs;
  result.children[1] = rhs;
  result.op = DOT;

  return result;
}

Tensor dg_exp(Tensor* in) {
  Tensor result;
  dg_InitTensor(&result, in->size);
  result.prev[0] = (double*)malloc(in->size * sizeof(double));
  for (size_t i = 0; i < in->size; ++i) {
    result.data[i] = exp(in->data[i]);
    result.prev[0][i] = in->data[i];
  }
  result.children[0] = in;
  result.op = EXP;

  return result;
}

Tensor dg_tanh(Tensor* in) {
  Tensor result;
  dg_InitTensor(&result, in->size);
  result.prev[0] = (double*)malloc(in->size * sizeof(double));
  for (size_t i = 0; i < in->size; ++i) {
    result.data[i] = tanh(in->data[i]);
    result.prev[0][i] = in->data[i];
  }
  result.children[0] = in;
  result.op = TANH;

  return result;
}

Tensor dg_relu(Tensor* in) {
  Tensor result;
  dg_InitTensor(&result, in->size);
  result.prev[0] = (double*)malloc(in->size * sizeof(double));
  for (size_t i = 0; i < in->size; ++i) {
    result.data[i] = in->data[i] > 0 ? in->data[i] : 0.0;
    result.prev[0][i] = in->data[i];
  }
  result.children[0] = in;
  result.op = RELU;

  return result;
}
