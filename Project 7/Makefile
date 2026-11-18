# Project 7 - matrix multiplication on a torus multicomputer
#
#   make            build the real-hardware programs (pthreads, CUDA)
#   make cstar      the C* simulator study (main submission; needs Wine)
#   make bench      the real-hardware study (CPU, GPU)
#   make util       CPU + GPU utilization during one run
#   make capture    render results/*.txt as PNG screen captures
#   make zip        build the submission zip
#   make clean

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -Wall
LDLIBS  ?= -lm

all: cpu_torus cuda_torus

cpu_torus: cpu_torus.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

cuda_torus: cuda_torus.cu
	nvcc -O3 -arch=native -std=c++17 -o $@ $< -lcublas

cstar:
	./benchmark.sh

bench: all
	./benchmark-hw.sh

util: all
	./utilization.sh

capture:
	./capture.sh

zip:
	./makezip.sh

clean:
	rm -f cpu_torus cuda_torus LISTFILE.TXT
	rm -rf build

.PHONY: all cstar bench util capture zip clean
