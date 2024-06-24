
all:
	clang++ -Wall -Wextra -g -o bin/prog src/main.cpp src/Tensor.cpp -std=c++14 -stdlib=libc++
