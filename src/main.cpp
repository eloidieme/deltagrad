#include <stdio.h>
#include <stdlib.h>

typedef struct Closure Closure;
typedef struct Value Value;

struct Value {
    double val;
    Value *children[2];
    char op;
    double grad;
    void (*repr)(Value);
    Closure *backward;
    Value (*add)(Value, Value);
};

struct Closure {
    Value *op1;
    Value *op2;
    void (*backward)(Closure *, Value);
};

void reprVal(Value self) {
    printf("Value(data=%g)\n", self.val);
}

void addBck(Closure *clos, Value out) {
    clos->op1->grad = 1.0 * out.grad;
    clos->op2->grad = 1.0 * out.grad;
}

Value addVals(Value op1, Value op2) {
    Closure _backward = {
        .op1 = &op1,
        .op2 = &op2,
        .backward = addBck};
    Value res = {.val = op1.val + op2.val, .children = {&op1, &op2}, .op = '+', .backward = &_backward, .repr = reprVal, .add = addVals};
    return res;
}

int main(int argc, char *argv[argc + 1]) {
    Value myVal = {.val = 50, .repr = reprVal, .add = addVals, .grad = 0.0};
    Value myVal2 = {.val = 20, .repr = reprVal, .add = addVals, .grad = 0.0};

    printf("val = %g\n", myVal.grad);
    Value newVal = myVal.add(myVal, myVal2);
    newVal.grad = 1.0;
    newVal.backward->backward(newVal.backward, newVal);
    newVal.repr(newVal);
    printf("val = %g\n", newVal.children[0]->grad);
    printf("op = %c\n", newVal.op);

    return EXIT_SUCCESS;
}
