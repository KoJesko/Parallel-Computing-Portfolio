# Project 6 - numerical integration on a 2-D mesh
#
#   make            build the MPI, pthreads and CUDA programs
#   make cstar      the C* simulator study (main submission; needs Wine)
#   make bench      the real-hardware study (MPI, pthreads, CUDA)
#   make capture    render results/*.txt as PNG screen captures
#   make clean

CC      ?= gcc
CFLAGS  ?= -O2 -march=native -Wall
LDLIBS  ?= -lm

all: mesh_integrate cpu_mesh cuda_mesh

mesh_integrate: mesh_integrate.c
	mpicc $(CFLAGS) -o $@ $< $(LDLIBS)

cpu_mesh: cpu_mesh.c
	$(CC) $(CFLAGS) -pthread -o $@ $< $(LDLIBS)

cuda_mesh: cuda_mesh.cu
	nvcc -O3 -arch=native -o $@ $<

cstar:
	./benchmark.sh

bench: all
	./benchmark-hw.sh

capture:
	./capture.sh

clean:
	rm -f mesh_integrate cpu_mesh cuda_mesh LISTFILE.TXT
	rm -rf build

.PHONY: all cstar bench capture clean
