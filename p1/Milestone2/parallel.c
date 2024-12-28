#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Constants are read in as input from command line

int prev(int i, int N);
int next(int i, int N);
void initialize(int N, double *C, double L, double dx, double dy);
void updateGrid(int N, double *C, double *Cnext);
void write_to_file(int N, double *C, FILE *file);


// Function to initialize the Gaussian pulse
void initialize(int N, double *C, double L, double dx, double dy) {
    double center_x = L / 2;
    double center_y = L / 2;

    //parallelizing the loop
    #ifdef _OPENMP
    #pragma omp parallel for default(none) shared(C, N, L, dx, dy, center_x, center_y)
    #endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * dx - center_x;
            double y = j * dy - center_y;
            double val = exp(-100 * (x * x + y * y));
            // printf("%f \n", val);
            C[i * N + j] = val;
        }
    }
}

void updateGrid(int N, double *C, double *Cnext) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = Cnext[i * N + j];
        }
    }
}

void write_to_file(int N, double *C, FILE *file) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%f ", C[i * N + j]);
        }
        fprintf(file, "\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s N NT L T u v nt\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    //malloc
    //const int NT = atoi(argv[2]);
    const double L = atof(argv[2]);
    const double T = atof(argv[3]);
    const double u = atof(argv[4]);
    const double v = atof(argv[5]);
    const int nt = atoi(argv[6]);
    const double dx = L / (N - 1);
    const double dy = L / (N - 1);
    const double dt = 0.5 * (dx / sqrt(pow(u,2)+pow(v,2)));

    double grindRate;
    
    int NT = (T/dt);
    printf("NT %d\n", NT);    
    
    int iprev;
    int inext;
    int jprev;
    int jnext;

    int iprev2;
    int inext2;
    int jprev2;
    int jnext2;

    double *C, *Cnext;

    
    // Allocate memory for C and Cnext
    C = (double *)calloc(N * N, sizeof(double));
    Cnext = (double *)calloc(N * N, sizeof(double));

    // Check for successful allocation
    if (!C || !Cnext) {
        fprintf(stderr, "Memory allocation failed\n");
        // Free allocated memory and exit if any allocation failed
        free(C);
        free(Cnext);
        return 1;
    }


    #ifdef _OPENMP
    double runtime = omp_get_wtime(); 
    omp_set_num_threads(nt); /*Sets nymber of threads to input value*/
    #endif

    if (dt > dx/sqrt(2*(u*u+v*v))){
		exit(1);
	}

    long long int gridcellNum = 0;

    initialize(N, C, L, dx, dy);

    char *filename = "output.txt";
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }
    
   // for (int n = 0; n < NT; n++) {
   
    for (int n = 0; n < NT; n++) {
        
        //parallelizing the loop
        #ifdef _OPENMP
        #pragma omp parallel for default(none) shared(C, Cnext) \
        private(iprev, inext, jprev, jnext, iprev2, inext2, jprev2, jnext2) \
        //reduction(+:gridcellNum)

        #endif

        #ifdef LAX
        //lax method
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int iprev = (i + 1) % N;
                int inext = (i - 1 + N) % N;
                int jprev = (j + 1) % N;
                int jnext = (j - 1 + N) % N;

                // int iprev = prev(i, N);
                // int inext = next(i, N);
                // int jprev = prev(j, N);
                // int jnext = next(j, N);

                // Lax scheme update
                Cnext[i * N + j] = 0.25 * (C[iprev * N + jprev] + C[inext * N + jprev] + C[iprev * N + jnext] + C[inext * N + jnext])
                             - 0.5 * (((u * dt / dx) * (C[iprev * N + j] - C[inext * N + j]))
                                      + ((v * dt / dy) * (C[i * N + jprev] - C[i * N + jnext])));            
            }
        }

        #elif FIRST_ORDER
        //first order method
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                iprev = (i + 1) % N;
                inext = (i - 1 + N) % N;
                jprev = (j + 1) % N;
                jnext = (j - 1 + N) % N;

                // case1: v > 0
                // upwind scheme update:
                
                if (u > 0 && v > 0){ 
                    Cnext[i * N + j] = C[i * N + j]
                                     - (dt/dx * (u*(C[i * N + j] - C[inext * N + j])))
                                     - (dt/dy * (v*(C[i * N + j] - C[i * N + jnext])));
                }
                else if (u < 0 && v < 0) {
                     Cnext[i * N + j] =  C[i * N + j] 
                                        - (dt/dx * (u*(C[iprev * N + j] - C[i * N + j])))
                                        - (dt/dy * (v*(C[i * N + jprev] - C[i * N + j])));
                }
                
                /*
                if (u > 0 && v > 0){ 
                    Cnext[i * N + j] = C[i * N + j] - (dt/dx * ((u*(C[i * N + j] - C[inext * N + j])) + (v*(C[i * N + j] - C[i * N + jnext]))));
                }
                else if (u < 0 && v < 0) {
                     Cnext[i * N + j] =  C[i * N + j] - (dt/dx * ((u*(C[iprev * N + j] - C[i * N + j])) + (v*(C[i * N + jprev] - C[i * N + j]))));
                }
                */

            }
        }

        #else 
        //second order method
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                iprev = (i + 1) % N;
                inext = (i - 1 + N) % N;
                jprev = (j + 1) % N;
                jnext = (j - 1 + N) % N;
                iprev2 = (i + 2) % N;
                inext2 = (i - 2 + N) % N;
                jprev2 = (j + 2) % N;
                jnext2 = (j - 2 + N) % N;

                // case1: v > 0
                // upwind scheme update:
                if (u > 0 && v > 0){
                    Cnext[i * N + j] = C[i * N + j]
                                    - ((0.5 * dt)/(2 * dx)) * (u * (3 * C[i * N + j] - 4 * C[inext * N + j] + C[inext2 * N + j]))
                                    - ((0.5 * dt)/(2 * dy)) * (v * (3 * C[i * N + j] - 4 * C[i * N + jnext] + C[i * N + jnext2]));
                }
                else if (u < 0 && v < 0) {
                    Cnext[i * N + j] = C[i * N + j] 
                                    - ((0.5 * dt)/(2 * dx)) * (u * (-1 * C[iprev2 * N + j] + 4 * C[iprev * N + j] - 3 * C[i * N + j])) 
                                    - ((0.5 * dt)/(2 * dy)) * (v * (-1 * C[i * N + jprev2] + 4 * C[i * N + jprev] - 3 * C[i * N + j]));
                                    
                }
            }
        }
        #endif
        updateGrid(N, C, Cnext);
        gridcellNum += (long long int)N * N;
        write_to_file(N, C, file);

    }

    #ifdef _OPENMP
    runtime = omp_get_wtime() - runtime;
    printf("time(s): %f\n", runtime);
    grindRate = (double)gridcellNum / runtime;

    //grindRate = (N * N) / runtime;
    printf("grindrate: %f\n", grindRate);

    #endif

    // write_to_file(N, C, "FO_output.txt");
    fclose(file);

    free(C);
    free(Cnext);

    return 0;
}

int prev(int i, int N) {
    return i == 0 ? N - 1 : i - 1;
}

int next(int i, int N) {
    return i == N - 1 ? 0 : i + 1;
}
