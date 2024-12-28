#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Constants are read in as input from command line

int prev(int i, int N);
int next(int i, int N);
void initialize(int N, double (*C)[N], double L);
void lax(int N, double (*C)[N], double (*Cnext)[N], double u, double v, double dt, double dx);
void updateGrid(int N, double (*C)[N], double (*Cnext)[N]);
void write_to_file(int N, const double (*C)[N], FILE *file);



// Function to initialize the Gaussian pulse
void initialize(int N, double (*C)[N], double L) {
    double dx = L / N;
    double dy = L / N;
    double x0 = L / 2;
    double y0 = L / 2;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = i * dx;
            double y = j * dy;
            double val = pow((x - x0), 2) / (2 * pow(L / 4, 2)) + pow((y - y0), 2) / (2 * pow(L / 4, 2));
            C[i][j] = exp(-val);
        }
    }
}

void updateGrid(int N, double (*C)[N], double (*Cnext)[N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = Cnext[i][j];
        }
    }
}

void write_to_file(int N, const double (*C)[N], FILE *file) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(file, "%f ", C[i][j]);
        }
        fprintf(file, "\n");
    }
}

int main(int argc, char *argv[]) {
    if (argc != 7) {
        fprintf(stderr, "Usage: %s N NT L T u v\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    const int NT = atoi(argv[2]);
    const double L = atof(argv[3]);
    const double T = atof(argv[4]);
    const double u = atof(argv[5]);
    const double v = atof(argv[6]);
    const double dx = L / N;
    const double dt = T / NT;
    int iprev;
    int inext;
    int jprev;
    int jnext;

    double C[N][N], Cnext[N][N];

    if (dt > dx/sqrt(2*(u*u+v*v))){
		exit(1);
	}

    initialize(N, C, L);

    char *filename = "output.txt";
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        exit(1);
    }

    for (int n = 0; n < NT; n++) {
        if( !(n%100) ) 
            write_to_file(N, C, file);

        //lax method
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                iprev = prev(i, N);
                inext = next(i, N);
                jprev = prev(j, N);
                jnext = next(j, N);
                Cnext[i][j] = 0.25 * (C[iprev][j] + C[inext][j] + C[i][jprev] + C[i][jnext])
                    - dt / (2 * dx) * (u * (C[inext][j] - C[iprev][j]) + v * (C[i][jnext] - C[i][jprev]));
            }
        }
        updateGrid(N, C, Cnext);
    }

    fclose(file);

    return 0;
}

int prev(int i, int N) {
    return i == 0 ? N - 1 : i - 1;
}

int next(int i, int N) {
    return i == N - 1 ? 0 : i + 1;
}

