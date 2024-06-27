#include <stdlib.h>

#include "Tensor.h"

#define N 6

int main(void) {
  // Initialization
  Tensor X, Y, W, b;
  dg_InitTensor(&X, 1);
  dg_InitTensor(&Y, 1);
  dg_InitTensor(&W, 1);
  dg_InitTensor(&b, 1);
  double dataX[N] = {0, 1, 2, 3, 4, 5};
  double dataY[N] = {0, 2, 4, 6, 8, 10};
  double dataW[1] = {4.2};
  double datab[1] = {1.2};
  double eta = 0.01;

  Tensor prod, h, error, loss, n, L;
  dg_InitTensor(&n, 1);
  n.data[0] = 1;
  W.data[0] = dataW[0];
  b.data[0] = datab[0];

  for (size_t k = 0; k < 100; ++k) {
    for (size_t i = 0; i < N; ++i) {
      X.data[0] = dataX[i];
      Y.data[0] = dataY[i];

      // Forward pass
      prod = dg_mult(&W, &X);
      h = dg_add(&prod, &b);
      error = dg_subs(&h, &Y);
      loss = dg_dot(&error, &error);
      L = dg_div(&loss, &n);

      // Backward pass
      dg_backward(&L);

      // Update
      W.data[0] -= eta * W.grad[0];
      b.data[0] -= eta * b.grad[0];
      W.grad[0] = 0.0;
      b.grad[0] = 0.0;
    }
    dg_repr("Loss", &L);
  }

  // Clean up
  dg_DestroyTensor(&X);
  dg_DestroyTensor(&Y);
  dg_DestroyTensor(&W);
  dg_DestroyTensor(&b);
  dg_DestroyTensor(&n);
  dg_DestroyTensor(&prod);
  dg_DestroyTensor(&h);
  dg_DestroyTensor(&error);
  dg_DestroyTensor(&loss);
  dg_DestroyTensor(&L);

  return EXIT_SUCCESS;
}
