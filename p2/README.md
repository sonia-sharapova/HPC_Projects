# Project 2

### Command Line Parameters:
#### 1: Number of rays
#### 2: Dimensions of square grid (n)
#### 3: Number of blocks
#### 4: Number of threads per block


## Compilation: 
### For Quadro (RX6000):
use -arch=sm_75

### For V100:
use -arch=sm_65

#### For single precision code:
nvcc -Xcompiler -fopenmp -O3 -use_fast_math -o raytrace raytrace.cu -arch=sm_?

#### For double precision code:
nvcc -Xcompiler -fopenmp -O3 -use_fast_math -o double double_p.cu -arch=sm_?


## To Run:
#### Double precision:

./double 1000000000 1000 1000 512

#### Single Precision:

./raytrace 1000000000 1000 1000 512




