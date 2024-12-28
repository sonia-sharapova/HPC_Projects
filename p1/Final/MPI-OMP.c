#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

MPI_Comm comm;

int prev(int i, int N);
int next(int i, int N);
void initialize(int N,  double L, double *C, double dx, FILE* filename);
void updateGrid(int N, double *C, double *Cnext);
void updateGridFinal(int N, int n, int start_i, int end_i, int start_j, int end_j, double *Cnew, double *Csub);

void write_to_file(int N, double *C, FILE *file);


void printArray(double *C_n, int N_i, int N_j);
void invokeMPI(int mype, int row, int col, int north, int south, int east, int west, int n, double *C, double **ghosts);
void getSubC(int start_i, int end_i, int start_j, int end_j, int row, int col, double *C_Sub, double *C_Final, int n, int k);


int N = 4000, TH, num_threads = 1;
double const L = 1.0, T = 1.0;
double dx, dt, NT;
double start_time, end_time;

// Function for non-divisibility


// Function to initialize the Gaussian pulse
void initialize(int N,  double L, double *C, double dx, FILE* filename) {
    double center_x = 1.0 / 2;
    double center_y = 1.0 / 2;

    double c = 0.0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * dx - center_x;
            double y = j * dx - center_y;
            double val;
    
            if (-0.1 <= y && y <= 0.1){
                val = 1.0;
            }else{
                val = 0.0;
            }
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

void updateGridFinal(int N, int n, int start_i, int end_i, int start_j, int end_j, double *C_Final, double *C_Sub) {
    for (int i = start_i; i < end_i; i++) {
        for (int j = start_j; j < end_j; j++) {
            int ind_i = i - start_i + 1;
            int ind_j = j - start_j + 1;

            C_Final[i * N + j] = C_Sub[ind_i * (n+2) + ind_j];
            
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


void getSubC(int start_i, int end_i, int start_j, int end_j, int row, int col, double *C_Sub, double *C_Final, int n, int k){

    for (int i = start_i; i < end_i; i++) {
        for (int j = start_j; j < end_j; j++) {
            int ind_i = i - start_i + 1;
            int ind_j = j - start_j + 1;

            //printf("ind_i:%d, ind_j%d:\n", ind_i, ind_j);
            C_Sub[ind_i * (n+2) + ind_j] = C_Final[i * N + j];
        }
    }


}

void invokeMPI(int mype, int row, int col, int north, int south, int east, int west, int n, double *C, double **ghosts) {

    double *ghost_n_recv = ghosts[0];
    double *ghost_e_recv = ghosts[1];
    double *ghost_s_recv = ghosts[2];
    double *ghost_w_recv = ghosts[3];

    double *ghost_n_send = ghosts[4];
    double *ghost_e_send = ghosts[5];
    double *ghost_s_send = ghosts[6];
    double *ghost_w_send = ghosts[7];
    
    for (int i = 1; i < n+1; i++){
        
        ghost_n_send[i-1] = C[(n+2)+i];
        ghost_s_send[i-1] = C[(n+2)*(n+1) - (n+2) + i];
        ghost_w_send[i-1] = C[i * (n+2)+1];
        ghost_e_send[i-1] = C[i * (n+2) + n];

    }

    //Send out grid data
    // even columns send east
    if (col % 2 == 0) {
        MPI_Send(ghost_e_send, n, MPI_DOUBLE, east, 0, MPI_COMM_WORLD);
        //printf("row %d col %d SENT EAST\n", row, col);
    }
    // odd columns receive from west 
    if (col % 2 == 1) {
        MPI_Recv(ghost_w_recv, n, MPI_DOUBLE, west, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV EAST\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrie: even col sent eastr\n");

    if (col % 2 == 0) {
        MPI_Send(ghost_w_send, n, MPI_DOUBLE, west, 1, MPI_COMM_WORLD);
        //printf("row %d col %d SENT WEST\n", row, col);
    }
    // odd columns receive from west 
    if (col % 2 == 1) {
        MPI_Recv(ghost_e_recv, n, MPI_DOUBLE, east, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV WEST\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: even col sent west \n");

    if (col % 2 == 1) {
        MPI_Send(ghost_e_send, n, MPI_DOUBLE, east, 2, MPI_COMM_WORLD);
        //printf("row %d col %d SENT EAST\n", row, col);
    }
    // odd columns receive from west 
    if (col % 2 == 0) {
        MPI_Recv(ghost_w_recv, n, MPI_DOUBLE, west, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV EAST\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: odd col sent east \n");

    if (col % 2 == 1) {
        MPI_Send(ghost_w_send, n, MPI_DOUBLE, west, 3, MPI_COMM_WORLD);
        //printf("row %d col %d SENT WEST\n", row, col);
    }
    // odd columns receive from west 
    if (col % 2 == 0) {
        MPI_Recv(ghost_e_recv, n, MPI_DOUBLE, east, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV WEST\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: sent and recieved over cols \n");




    //Send out grid data
    // even columns send north
    if (row % 2 == 0) {
        MPI_Send(ghost_n_send, n, MPI_DOUBLE, north, 4, MPI_COMM_WORLD);
        //printf("row %d row %d SENT NORTH\n", row, col);
    }
    // odd columns receive from north 
    if (row % 2 == 1) {
        MPI_Recv(ghost_s_recv, n, MPI_DOUBLE, south, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d row %d RCV NORTH\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrie: even row sent north\n");

    if (row % 2 == 0) {
        MPI_Send(ghost_s_send, n, MPI_DOUBLE, south, 5, MPI_COMM_WORLD);
        //printf("row %d col %d SENT SOUTH\n", row, col);
    }
    // odd rows receive from west 
    if (row % 2 == 1) {
        MPI_Recv(ghost_n_recv, n, MPI_DOUBLE, north, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV SOUTH\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: even row sent south \n");

    if (row % 2 == 1) {
        MPI_Send(ghost_n_send, n, MPI_DOUBLE, north, 6, MPI_COMM_WORLD);
        //printf("row %d col %d SENT NORTH\n", row, col);
    }
    // odd rows receive from north 
    if (row % 2 == 0) {
        MPI_Recv(ghost_s_recv, n, MPI_DOUBLE, south, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV NORTH\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: odd row sent north \n");

    if (row % 2 == 1) {
        MPI_Send(ghost_s_send, n, MPI_DOUBLE, south, 7, MPI_COMM_WORLD);
        //printf("row %d col %d SENT SOUTH\n", row, col);
    }
    // odd rows receive from south 
    if (row % 2 == 0) {
        MPI_Recv(ghost_n_recv, n, MPI_DOUBLE, north, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //printf("row %d col %d RCV SOUTH\n", row, col);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //printf("After barrier: signals sent over rows \n");


    for (int i = 1; i < n+1; i++){

        C[i] = ghost_n_recv[i-1];
        C[(n+2)*(n+2) - (n+2) + i] = ghost_s_recv[i-1];
        C[i * (n+2)] = ghost_w_recv[i-1];
        C[i * (n+2) + (n + 1)] = ghost_e_recv[i-1];

    }
}

int main(int argc, char *argv[]) {
    dx = L/(N-1);
    dt = 0.000125;
    NT = (double) T/dt;

    //NT = 5000;

    int nprocs; /* number of processes used in this invocation */
    int mype  ; /* my processor id (from 0 .. nprocs-1) */
    int size, rank, dims[2], periods[2] = {0, 0}, reorder = 1, coords[2];

    MPI_Status stat;

    double *C_Final, *C_Sub, *C_SubNext, *C_Dirs;

    // Allocate memory for C and Cnext
    
    C_Final = (double *)calloc(N * N, sizeof(double));


    char *filename = "output.txt";
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    MPI_Init(&argc, &argv);  /* do this first to init MPI */
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);  
    
    
    /* Initialize */
    printf("MPI processes: %d\n", mype);

    initialize(N, L, C_Final, dx, file);


    int k = sqrt(nprocs);
    int n = N / k;

    C_Sub = (double *)calloc((n+2) * (n+2), sizeof(double));
    C_SubNext = (double *)calloc((n+2) * (n+2), sizeof(double));
    
    // for indices
    int row = mype / k;  //Nx
    int col = mype % k;  //Ny

    // int nl = (N * N)/nprocs;  /* size of subarray on each proc */
    int start_i = row * n;
    int end_i = start_i + n;
    int start_j = col * n;
    int end_j = start_j + n;

    // Identify neighbors
    int north = (row == 0) ? mype + (k - 1)*(k) : mype - k;
    int east = (col == k - 1) ? mype - (k - 1) : mype + 1;
    int south = (row == k - 1) ? mype - (k - 1)*(k) : mype + k;
    int west = (col == 0) ? mype + (k - 1) : mype - 1;

    getSubC(start_i, end_i, start_j, end_j, row, col, C_Sub, C_Final, n, k);

    omp_set_num_threads(num_threads); 
    double start_time = omp_get_wtime();

    double *ghosts[8]; 
    for (int i=0; i<8; ++i)
        ghosts[i] = (double *)calloc(n, sizeof(double));

    for(int t_s=0; t_s < NT; t_s++){
        //if(mype==0) printf("Iteration %d\n", t_s);
        invokeMPI(mype, row, col, north, south, east, west, n, C_Sub, ghosts);
        //printf("CDIRS\n");
        //printArray(C_Dirs, 4, n);

        #pragma omp parallel for collapse(2) default(none) \
        shared(C_Sub, C_SubNext, N, dt, dx, start_i, start_j, k, n, row, col) 
        //private(x, y, u, v)
        for (int i = 1; i < n+1; i++) {
            for (int j = 1; j < n+1; j++) {


                int global_i = row*n + i - 1;
                int global_j = col*n + j - 1;


                double x = -L/2 + dx * global_i;
                double y = -L/2 + dx * global_j;
                double u = sqrt(2) * y;
                double v = -(sqrt(2)) * x;


                int val_iprev = i + 1;
                int val_inext = i - 1;
                int val_jprev = j + 1;
                int val_jnext = j - 1;


                C_SubNext[i * (n+2) + j] = (C_Sub[val_iprev*(n+2) + j] + C_Sub[val_inext*(n+2) + j] + C_Sub[i*(n+2) + val_jprev] + C_Sub[i*(n+2) + val_jnext])/4
                                -(dt/(2*dx))*(u*(C_Sub[val_inext*(n+2) + j] - C_Sub[val_iprev*(n+2) + j]) + v*(C_Sub[i*(n+2) + val_jnext] - C_Sub[i*(n+2) + val_jprev]));


            }
        }


        updateGrid(n+2, C_Sub, C_SubNext);
    }

    for (int i=0; i<8; ++i)
        free(ghosts[i]);

    if(mype == 0){

        updateGridFinal(N, N/k, start_i, end_i, start_j, end_j, C_Final, C_Sub);

        /* Send out data to the workers */
        for(int pid = 1; pid<nprocs; pid++){

            //printf("HELLOO");

            MPI_Recv(C_Sub, (n+2)*(n+2), MPI_DOUBLE, pid, 4, MPI_COMM_WORLD,&stat);
            
            int row_pe = pid / k;  //Nx
            int col_pe = pid % k;  //Ny
            int start_i_pe = row_pe * n;
            int end_i_pe = (row_pe == k - 1) ? (N) : (start_i_pe + n);
            int start_j_pe = col_pe * n;
            int end_j_pe = (col_pe == k - 1) ? (N) : (start_j_pe + n);

            //getSubC(start_i, end_i, start_j, end_j, row, col, C_Sub, C_Final, n, k);
            

            updateGridFinal(N, N/k, start_i_pe, end_i_pe, start_j_pe, end_j_pe, C_Final, C_Sub);
        }
    }
    else{
        MPI_Send(C_Sub, (n+2)*(n+2), MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
    
    if (mype == 0){
        //printArray(C_Final, N, N);
        write_to_file(N, C_Final, file);
    }


    fclose(file);
    //printArray(C_Final, N, N);


    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    printf("Elapsed Time: %f seconds \n", elapsed_time);

    free(C_Sub);
    free(C_SubNext);
    free(C_Final);


    MPI_Finalize();

}

int prev(int i, int N) {
    return i == 0 ? N - 1 : i - 1;
}

int next(int i, int N) {
    return i == N - 1 ? 0 : i + 1;
}


void printArray(double *C_n, int N_i, int N_j) {
    for (int i = 0; i < N_i; i++) {
        for (int j = 0; j < N_j; j++) {
            printf("%f ", C_n[i * N_i + j]);
        }
        printf("\n");
    }
}



