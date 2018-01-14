/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <vector>

//compute the matrix index for the ith row and jth column of a matrix with a given number of rows
const int get_index(const int i, const int j, const int number_of_rows)
{
  return i*number_of_rows + j;
}

//compute the l2 norm of a vector
const double compute_l2_norm(const std::vector<double> &x)
{
  double l2 = 0.;
  int n = x.size();
  for(int i = 0; i < n; ++i) l2 += x[i]*x[i];
  return sqrt(l2);
}

// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
  for(int i = 0; i < n; ++i) y[i] = 0.;
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      y[i] += A[get_index(i, j, n)]*x[j];
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
  for(int i = 0; i < n; ++i) y[i] = 0.;
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < m; ++j)
      y[i] += A[get_index(i, j, n)]*x[j];
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
  //initialize x to 0
  for(int i = 0; i < n; ++i) x[i] = 0.;

  //define inverse diagonal of A (as vector for efficiency)
  std::vector<double> D_inv(n);
  for(int i = 0; i < n; ++i) D_inv[i] = (A[get_index(i, i, n)] == 0.) ? 0 : 1./A[get_index(i, i, n)];

  //define matrix for non-diagonal terms of A
  std::vector<double> R(n*n);
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      R[get_index(i, j, n)] = (i == j) ? 0. : A[get_index(i, j, n)];

  //perform iterations to solve the system using Jacobi method
  std::vector<double> Rx(n);
  std::vector<double> Ax(n);
  std::vector<double> Ax_minus_b(n);
  double l2_error = 1.e10;
  for(int iterations_completed = 0; iterations_completed < max_iter && l2_error > l2_termination; ++iterations_completed)
  {
    //update x
    matrix_vector_mult(n, &R[0], x, &Rx[0]);
    for(int i = 0; i < n; ++i) x[i] = D_inv[i]*(b[i] - Rx[i]);

    //compute l2 error
    matrix_vector_mult(n, &A[0], x, &Ax[0]);
    for(int i = 0; i < n; ++i) Ax_minus_b[i] = Ax[i] - b[i];
    l2_error = compute_l2_norm(Ax_minus_b);
  }
}
