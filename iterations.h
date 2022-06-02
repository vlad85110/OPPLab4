#pragma once

#define X_0 -1
#define Y_0 -1
#define Z_0 -1
#define D 2
#define A 100000
#define EPS 0.00000001
#define F_0 0

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double function_fi(double x, double y, double z);

double function_ro(double x, double y, double z);

double h_x(double N);

double h_y(double N);

double h_z(double N);

double x_i(double i, double N);

double y_i(double i, double N);

double z_i(double i, double N);

double next_fi(int i, int j, int k, int N, const double *prev, int N_z);

double upper_next_fi(int i, int j, int k, int N, const double *prev, const double *upper, int N_z);

double lower_next_fi(int i, int j, int k, int N, const double *prev, const double *lower, int N_z);

void iterations();
