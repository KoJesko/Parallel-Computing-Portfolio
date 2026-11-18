# Project 5 - parallel Jacobi relaxation
#
#   make            build the OpenMP program for this machine
#   make run        a sample run on all cores
#   make bench      the full real-hardware performance study
#   make cstar      the C* simulator study (needs Wine for cstar.exe)
#   make clean

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -fopenmp -std=c11 -Wall -Wextra
LDLIBS  ?= -lm

all: JacobiOpenMP

JacobiOpenMP: JacobiOpenMP.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

run: JacobiOpenMP
	./JacobiOpenMP -n 1024 -t 0.01 -L

bench: JacobiOpenMP
	./benchmark-openmp.sh

cstar:
	./benchmark.sh

clean:
	rm -f JacobiOpenMP
	rm -rf build

.PHONY: all run bench cstar clean
