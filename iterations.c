#include "iterations.h"

double abs_d(double num) {
    return num >= 0 ? num:-num;
}

double function_fi(double x, double y, double z){
    return pow(x, 2) + pow(y, 2) + pow(z, 2);
}

double function_ro(double x, double y, double z) {
    return 6 - A * function_fi(x, y ,z);
}

double h_x(double N_x) {
    return D / (N_x - 1);
}

double h_y(double N_y) {
    return D / (N_y - 1);
}

double h_z(double N_z) {
    return D / (N_z - 1);
}

double x_i(double i, double N) {

    return X_0 + i * h_x(N);
}

double y_i(double i, double N) {
    return Y_0 + i * h_y(N);
}

double z_i(double i, double N) {
    return Z_0 + i * h_z(N);
}

double upper_next_fi(int i, int j, int k, int N, const double *prev, const double *upper, int N_z) {
    double h_x2 = pow(h_x(N), 2);
    double h_y2 = pow(h_y(N), 2);
    double h_z2 = pow(h_z(N), 2);
    double _const = 1 / ((2 / h_x2) + (2 / h_y2) + (2 / h_z2) + A);

    double first = (prev[N * N_z * (i + 1) + N_z * j + k] +
                    prev[N * N_z * (i - 1) + N_z * j + k]) / h_x2;
    double second = (prev[N * N_z * i + N_z * (j + 1) + k] +
                     prev[N * N_z * i + N_z * (j - 1) + k]) / h_y2;
    double third = (upper[N * i + j] +
                    prev[N * N_z * i + N_z * j + (k - 1)]) / h_z2;

    return _const * (first + second + third - function_ro(x_i(i, N), y_i(j, N), z_i(k, N_z)));
}

double lower_next_fi(int i, int j, int k, int N, const double *prev, const double *lower, int N_z) {
    double h_x2 = pow(h_x(N), 2);
    double h_y2 = pow(h_y(N), 2);
    double h_z2 = pow(h_z(N), 2);
    double _const = 1 / ((2 / h_x2) + (2 / h_y2) + (2 / h_z2) + A);

    double first = (prev[N * N_z * (i + 1) + N_z * j + k] +
                    prev[N * N_z * (i - 1) + N_z * j + k]) / h_x2;
    double second = (prev[N * N_z * i + N_z * (j + 1) + k] +
                     prev[N * N_z * i + N_z * (j - 1) + k]) / h_y2;
    double third = (prev[N * N_z * i + N_z * j + (k + 1)] +
                    lower[N * i + j]) / h_z2;

    return _const * (first + second + third - function_ro(x_i(i, N), y_i(j, N), z_i(k, N_z)));
}

double next_fi(int i, int j, int k, int N, const double *prev, int N_z) {
    double h_x2 = pow(h_x(N), 2);
    double h_y2 = pow(h_y(N), 2);
    double h_z2 = pow(h_z(N), 2);
    double _const = 1 / ((2 / h_x2) + (2 / h_y2) + (2 / h_z2) + A);

    double first = (prev[N * N_z * (i + 1) + N_z * j + k] +
            prev[N * N_z * (i - 1) + N_z * j + k]) / h_x2;
    double second = (prev[N * N_z * i + N_z * (j + 1) + k] +
            prev[N * N_z * i + N_z * (j - 1) + k]) / h_y2;
    double third = (prev[N * N_z * i + N_z * j + (k + 1)] +
            prev[N * N_z * i + N_z * j + (k - 1)]) / h_z2;

    return _const * (first + second + third - function_ro(x_i(i, N), y_i(j, N), z_i(k, N_z)));
}

void fill_border(int N, double* array) {
    for (int i = 0; i < N; i += N - 1) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                array[N * N * i + N * j + k] = function_fi(x_i(i , N), y_i(j, N), z_i(k, N));
            }
        }
    }

    for (int j = 0; j < N; j += N - 1) {
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < N; ++k) {
                array[N * N * i + N * j + k] = function_fi(x_i(i , N), y_i(j, N), z_i(k, N));
            }
        }
    }

    for (int k = 0; k < N; k += N - 1) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                array[N * N * i + N * j + k] = function_fi(x_i(i , N), y_i(j, N), z_i(k, N));
            }
        }
    }
}

void iterations() {
    MPI_Request send_request1, send_request2, recv_request1, recv_request2;
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8;
    int sub_N = N / size;
    double max = F_0;
    double res, pr;

    double *prev, *sub_prev, *upper, *lower;

    sub_prev = (double *)calloc(sub_N * N * N, sizeof(double));
    upper = (double *)calloc(N * N, sizeof(double));
    lower = (double *)calloc(N * N, sizeof(double));

    if (rank == 0) {
        prev = (double *) calloc(N * N * N, sizeof(double));
        fill_border(sub_N, prev);
    }

    MPI_Scatter(prev, sub_N * N * N, MPI_DOUBLE,
                sub_prev, sub_N * N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    do {
        if (rank != 0) {
            MPI_Isend(sub_prev, N * N, MPI_DOUBLE, rank - 1,
                      124, MPI_COMM_WORLD, &send_request1);

            MPI_Irecv(lower, N * N, MPI_DOUBLE, rank - 1,
                      124, MPI_COMM_WORLD, &recv_request1);
        }

        if (rank != size - 1) {
            MPI_Isend(sub_prev + ((sub_N - 1) * N * N), N * N, MPI_DOUBLE, rank + 1,
                      123, MPI_COMM_WORLD, &send_request2);

            MPI_Irecv(upper, N * N, MPI_DOUBLE, rank + 1,
                      123, MPI_COMM_WORLD, &recv_request2);
        }

        max = -1;

        for (int k = sub_N * rank + 1, sub_k = 0; k < sub_N * (rank + 1) - 1; ++k, sub_k++) {
            for (int i = 1; i < N - 1; ++i) {
                for (int j = 1; j < N - 1; ++j) {
                    res = next_fi(i, j, k, N, prev, sub_N);

                    pr = (res - prev[N * N * sub_k + N * i + j]);
                    if (pr > max) {
                        max = pr;
                    }

                    prev[N * N * sub_k + N * i + j] = res;
                }
            }
        }

        fprintf(stderr, "%.10f\n", max);

        if (rank != 0) {
            MPI_Wait(&send_request1, MPI_STATUS_IGNORE);
        }

        if (rank != size - 1) {
            MPI_Wait(&send_request2, MPI_STATUS_IGNORE);
        }

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                res = lower_next_fi(i, j, sub_N * rank, N, prev, lower, sub_N);

                pr = (res - prev[N * i + j]);
                if (pr > max) {
                    max = pr;
                }

                prev[N * i + j] = res;
            }
        }

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                res = upper_next_fi(i, j, sub_N * (rank + 1), N, prev, upper, sub_N);

                pr = (res - prev[N * N * (sub_N - 1) + N * i + j]);
                if (pr > max) {
                    max = pr;
                }

                prev[N * N * (sub_N - 1) + N * i + j] = res;
            }
        }

    } while (max > EPS);

    //free(prev);
}
