all:
	g++ test.cpp -Wall -Wextra -std=c++0x -O3
run: all
	./a.out