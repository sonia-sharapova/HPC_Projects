cuda_blas: cuda_blas.o m_mult.o
	gcc cuda_blas.o m_mult.o -o cuda_blas -L/usr/local/cuda/lib64 -lcudart -lcublas -lopenblas -lm -O3

m_mult.o: m_mult.cu
	nvcc -c m_mult.cu -o m_mult.o

cuda_blas.o: cuda_blas.c
	gcc -c cuda_blas.c -o cuda_blas.o -O3

cpu_blas: blas_relu_nn.c
	gcc blas_relu_nn.c -o cpu_blas -O3 -lopenblas -lm

cpu_manual: manual_cpu.c
	gcc manual_cpu.c -o cpu_man -O3 -lm

load: 
	module load cuda
	module load openblas

run_cuda: cuda_blas
	 ./cuda_blas 1 800 50 200 0.1

run_cpu_blas: cpu_blas
	 ./cpu_blas 1 800 50 200 0.1

run_cpu_manual: cpu_man
	 ./cpu_man 1 800 50 200 0.1

clean:
	rm -f *.o cuda_blas cpu_blas
