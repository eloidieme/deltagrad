# Deltagrad

## Overview
This project provides a minimal automatic differentiation (autograd) engine implemented in C. The engine allows for basic tensor operations and automatic computation of gradients, essential for machine learning algorithms like gradient descent.
It was made to learn more about backprop in Deep Learning but also about the C programming language.

## Features
- Tensor operations including addition, subtraction, multiplication, division, and more.
- Automatic gradient computation for scalar-valued functions.

## Structure
- `Tensor.h`: Header file defining the `Tensor` structure and associated operations.
- `Tensor.c`: Implementation of tensor operations and autograd functionalities.
- `main.c`: Example usage of the autograd engine to perform computations.

## How to Build
Ensure you have a C compiler like GCC installed. Compile the project using the following command:
```bash
gcc src/main.c src/Tensor.c -o dgrad
```

## How to Run
After building the project, you can run it using:
```bash
./dgrad
```

## Example Usage
The `main.c` file contains an example of basic training to learn f: x -> 2*x. 
The loss is computed for successive epochs and can be seen decreasing as expected.

## Contributing
Contributions are welcome. Please feel free to fork the repository, make changes, and submit pull requests.

## License
This project is open-sourced under the MIT license. See the LICENSE file for more details.
