/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"
#include "mpi.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */


void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
  int q, i; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int recvcounts;
  int root;
  MPI_Comm column_comm;
  std::vector<double> localv;
  std::vector<int> sendcounts,displs;

  coords[0]=0;
  coords[1]=0;
  //get rank of (0, 0), which is used in MPI_Scatterv
  MPI_Cart_rank(comm, coords, &root);
  //determine value of q, get row and column number of each processor
  MPI_Cart_get(comm, maxdims, dims, periods, coords2);
  //create sub cart
  remain_dims[0]=1;
  remain_dims[1]=0;
  MPI_Cart_sub(comm, remain_dims, &column_comm);

  q=dims[0];
  sendcounts.resize(q);
  displs.resize(q);
  row_num=coords2[0];
  column_num=coords2[1];
  if(column_num==0){
    if(row_num<(n%q)){
      recvcounts=(int)ceil(double(n)/q);
      localv.resize(recvcounts);
    }
    else{
      recvcounts=(int)floor(double(n)/q);
      localv.resize(recvcounts);
    }
    for(i=0;i<q;i++){
      if(i<(n%q)){
	sendcounts[i]=(int)ceil(double(n)/q);
      }
      else{
	sendcounts[i]=(int)floor(double(n)/q);
      }
      if(i==0){
	displs[i]=0;
      }
      else{
	displs[i]=displs[i-1]+sendcounts[i-1];
      }
    }

    //If column==0, scatter the vector with vector sizes of floor(n/q) or ceil(n/q)
    MPI_Scatterv(input_vector, &sendcounts[0], &displs[0], MPI_DOUBLE, *local_vector, recvcounts, MPI_DOUBLE, root, column_comm);
  }
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
  int q, i; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int sendcounts;
  int root;
  MPI_Comm column_comm;
  std::vector<double> localv;
  std::vector<int> recvcounts,displs;

  coords[0]=0;
  coords[1]=0;
  //get rank of (0, 0), which is used in MPI_Scatterv
  MPI_Cart_rank(comm, coords, &root);
  //determine value of q, get row and column number of each processor
  MPI_Cart_get(comm, maxdims, dims, periods, coords2);
  //create sub cart
  remain_dims[0]=1;
  remain_dims[1]=0;
  MPI_Cart_sub(comm, remain_dims, &column_comm);

  q=dims[0];
  recvcounts.resize(q);
  displs.resize(q);
  row_num=coords2[0];
  column_num=coords2[1];
  if(column_num==0){
    if(row_num<(n%q)){
      sendcounts=(int)ceil(double(n)/q);
      localv.resize(sendcounts);
    }
    else{
      sendcounts=(int)floor(double(n)/q);
      localv.resize(sendcounts);
    }
    for(i=0;i<q;i++){
      if(i<(n%q)){
	recvcounts[i]=(int)ceil(double(n)/q);
      }
      else{
	recvcounts[i]=(int)floor(double(n)/q);
      }
      if(i==0){
	displs[i]=0;
      }
      else{
	displs[i]=displs[i-1]+recvcounts[i-1];
      }
    }
    //If column==0, scatter the vector with vector sizes of floor(n/q) or ceil(n/q)
    MPI_Gatherv(local_vector, sendcounts, MPI_DOUBLE, output_vector, &recvcounts[0], &displs[0], MPI_DOUBLE, root, column_comm);
  }
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
  int q, i, j, k; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int recvcounts;
  int root;
  int mark1,mark2,index1,index2;
  MPI_Comm column_comm, row_comm;
  std::vector<double> localv, localv2;
  std::vector<int>column_size, row_size;
  std::vector<int> sendcounts,displs;

  coords[0]=0;
  coords[1]=0;
  //get rank of (0, 0), which is used in MPI_Scatterv
  MPI_Cart_rank(comm, coords, &root);
  //determine value of q, get row and column number of each processor
  MPI_Cart_get(comm, maxdims, dims, periods, coords2);
  //create sub cart
  remain_dims[0]=1;
  remain_dims[1]=0;
  MPI_Cart_sub(comm, remain_dims, &column_comm);

  q=dims[0];
  sendcounts.resize(q);
  displs.resize(q);
  column_size.resize(q);
  row_size.resize(q);
  row_num=coords2[0];
  column_num=coords2[1];
  //First distribute within first column
  if(column_num==0){
    if(row_num<(n%q)){
      recvcounts=n*(int)ceil(double(n)/q);
      localv.resize(recvcounts);
      localv2.resize(recvcounts);
    }
    else{
      recvcounts=n*(int)floor(double(n)/q);
      localv.resize(recvcounts);
      localv2.resize(recvcounts);
    }
    for(i=0;i<q;i++){
      if(i<(n%q)){
	sendcounts[i]=n*(int)ceil(double(n)/q);
      }
      else{
	sendcounts[i]=n*(int)floor(double(n)/q);
      }
      if(i==0){
	displs[i]=0;
      }
      else{
	displs[i]=displs[i-1]+sendcounts[i-1];
      }
    }

    //If column==0, scatter the vector with vector sizes of floor(n/q) or ceil(n/q)
    MPI_Scatterv(input_matrix, &sendcounts[0], &displs[0], MPI_DOUBLE, &localv[0], recvcounts, MPI_DOUBLE, root, column_comm);
  }
  MPI_Barrier(comm);


  //Then distribute within rows
  remain_dims[0]=0;
  remain_dims[1]=1;
  MPI_Cart_sub(comm, remain_dims, &row_comm);
  if(row_num<(n%q)){
    if(column_num<(n%q)){
      recvcounts=(int)ceil(double(n)/q)*(int)ceil(double(n)/q);
    }
    else{
      recvcounts=(int)ceil(double(n)/q)*(int)floor(double(n)/q);
    }
    for(i=0;i<q;i++){
      if(i<(n%q)){
	row_size[i]=(int)ceil(double(n)/q);
	column_size[i]=(int)ceil(double(n)/q);
	sendcounts[i]=(int)ceil(double(n)/q)*(int)ceil(double(n)/q);
      }
      else{
	row_size[i]=(int)ceil(double(n)/q);
	column_size[i]=(int)floor(double(n)/q);
	sendcounts[i]=(int)ceil(double(n)/q)*(int)floor(double(n)/q);
      }
      if(i==0){
	displs[i]=0;
      }
      else{
	displs[i]=displs[i-1]+sendcounts[i-1];
      }
    }
  }
  else{
    if(column_num<(n%q)){
      recvcounts=(int)floor(double(n)/q)*(int)ceil(double(n)/q);
    }
    else{
      recvcounts=(int)floor(double(n)/q)*(int)floor(double(n)/q);
    }
    for(i=0;i<q;i++){
      if(i<(n%q)){
	row_size[i]=(int)floor(double(n)/q);
	column_size[i]=(int)ceil(double(n)/q);
	sendcounts[i]=(int)floor(double(n)/q)*(int)ceil(double(n)/q);
      }
      else{
	row_size[i]=(int)floor(double(n)/q);
	column_size[i]=(int)floor(double(n)/q);
	sendcounts[i]=(int)floor(double(n)/q)*(int)floor(double(n)/q);
      }
      if(i==0){
	displs[i]=0;
      }
      else{
	displs[i]=displs[i-1]+sendcounts[i-1];
      }
    }
  }
  
  //rearrange localv to localv2
  if(column_num==0){
    mark1=0;
    mark2=0;
    for (i=0;i<q;i++){
    
      for (j=0;j<row_size[i];j++){
	for (k=0;k<column_size[i];k++){
	  index1=mark1+j*column_size[i]+k;
	  index2=j*n+mark2+k;
	  localv2[mark1+j*column_size[i]+k]=localv[j*n+mark2+k];
	}
      }
      mark1=mark1+row_size[i]*column_size[i];
      mark2=mark2+column_size[i];
    }
  }


  MPI_Scatterv(&localv2[0], &sendcounts[0], &displs[0], MPI_DOUBLE, *local_matrix, recvcounts, MPI_DOUBLE, root, row_comm);

}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{

  int i, q; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int root;
  int size, rank_send, rank_recv;
  MPI_Comm column_comm;
  std::vector<double> localv;
  MPI_Status status;
  MPI_Request req1, req2;


  //determine value of q, get row and column number of each processor
  MPI_Cart_get(comm, maxdims, dims, periods, coords2);
  q=dims[0];
  row_num=coords2[0];
  column_num=coords2[1];

  if(row_num<(n%q)){
    size=(int)ceil(double(n)/q);
  }
  else{
    size=(int)floor(double(n)/q);
  }


  //First each processor (i, 0) sends its local vector to diagonal processor (i,i)
  if((column_num==0)&&(row_num==0)){
    for(i=0;i<size;i++){
      row_vector[i]=col_vector[i];
    }
  }

  if((column_num==row_num)&&(row_num!=0)){
    coords[0]=row_num;
    coords[1]=0;
    MPI_Cart_rank(comm, coords, &rank_send);
    MPI_Irecv(row_vector, size, MPI_DOUBLE, rank_send, 10, comm, &req2);
  }
     
  if((column_num==0)&&(row_num!=0)){
    coords[0]=row_num;
    coords[1]=row_num;
    //get rank of (i,i)
    MPI_Cart_rank(comm, coords, &rank_recv);
    MPI_Isend(col_vector, size, MPI_DOUBLE, rank_recv, 10, comm, &req1);
  }

  //create sub cart
  remain_dims[0]=1;
  remain_dims[1]=0;
  MPI_Cart_sub(comm, remain_dims, &column_comm);
  //find root for each row
  coords[0]=column_num;
  coords[1]=0;
  MPI_Cart_rank(column_comm, coords, &root);
    
  if(column_num<(n%q)){
    size=(int)ceil(double(n)/q);
  }
  else{
    size=(int)floor(double(n)/q);
  }

  MPI_Bcast(row_vector, size, MPI_DOUBLE, root, column_comm);

}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{

  int q, i, j; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int root;
  int column_size, row_size;
  MPI_Comm row_comm;
  std::vector<double> tempy;
  //double* local_x_new = new double[block_decompose_by_dim(n, comm, 0)];
  double* local_x_new = new double[block_decompose_by_dim(n, comm, 1)];

  //determine value of q, get row and column number of each processor
  MPI_Cart_get(comm, maxdims, dims, periods, coords2);

  q=dims[0];
  row_num=coords2[0];
  column_num=coords2[1];

  transpose_bcast_vector(n, local_x, local_x_new, comm);


  if(row_num<(n%q)){
    row_size=(int)ceil(double(n)/q);
  }
  else{
    row_size=(int)floor(double(n)/q);
  }

  if(column_num<(n%q)){
    column_size=(int)ceil(double(n)/q);
  }
  else{
    column_size=(int)floor(double(n)/q);
  }
  tempy.resize(row_size);


  for (j=0; j<row_size; j++){
    tempy[j]=0;
    local_y[j]=0;
    for (i=0; i<column_size; i++){
       tempy[j]=tempy[j]+local_A[j*column_size+i]*local_x_new[i];     
    }
  }
  

  //create sub cart
  remain_dims[0]=0;
  remain_dims[1]=1;
  MPI_Cart_sub(comm, remain_dims, &row_comm);
  //find root for each row
  coords[0]=0;
  coords[1]=0;
  MPI_Cart_rank(row_comm, coords, &root);

  MPI_Barrier(comm);

  MPI_Reduce(&tempy[0], &local_y[0], row_size, MPI_DOUBLE, MPI_SUM, root, row_comm);
}



// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
  int q, i, iter; //number of processors in first dimension of 2d grid communicator
  int remain_dims[2];
  int maxdims=2, dims[2]={0, 0}, periods[2]={0, 0},coords[2]={0, 0},coords2[2]={0, 0};
  int column_num, row_num;
  int root, signal=0;
  int column_size, row_size, size;
  MPI_Comm column_comm, row_comm;
  double l2_norm, l2;
  

  double* local_R = new double[block_decompose_by_dim(n, comm, 0)*block_decompose_by_dim(n, comm, 1)];
  double* local_D = new double[block_decompose_by_dim(n, comm, 0)];
  double* local_P = new double[block_decompose_by_dim(n, comm, 0)];
  double* local_w = new double[block_decompose_by_dim(n, comm, 0)];


  MPI_Cart_get(comm, maxdims, dims, periods, coords2);
  q=dims[0];
  row_num=coords2[0];
  column_num=coords2[1];

  if(row_num<(n%q)){
    if(column_num<(n%q)){
      row_size=(int)ceil(double(n)/q);
      column_size=(int)ceil(double(n)/q);
      size=(int)ceil(double(n)/q)*(int)ceil(double(n)/q);
    }
    else{
      row_size=(int)ceil(double(n)/q);
      column_size=(int)floor(double(n)/q);
      size=(int)ceil(double(n)/q)*(int)floor(double(n)/q);
    }
  }
  else{
    if(column_num<(n%q)){
      row_size=(int)floor(double(n)/q);
      column_size=(int)ceil(double(n)/q);
      size=(int)floor(double(n)/q)*(int)ceil(double(n)/q);
    }
    else{
      row_size=(int)floor(double(n)/q);
      column_size=(int)floor(double(n)/q);
      size=(int)floor(double(n)/q)*(int)floor(double(n)/q);
    }
  }

  for (i=0;i<size;i++){
    local_R[i]=local_A[i];
  }

  if(row_num==column_num){
    for(i=0;i<row_size;i++){
      local_D[i]=local_A[i*row_size+i];
      local_R[i*row_size+i]=0;
    }
  }
  //create sub cart
  remain_dims[0]=0;
  remain_dims[1]=1;
  MPI_Cart_sub(comm, remain_dims, &row_comm);
  //create sub cart
  remain_dims[0]=1;
  remain_dims[1]=0;
  MPI_Cart_sub(comm, remain_dims, &column_comm);

  //find root for each row
  coords[0]=row_num;
  coords[1]=0;
  MPI_Cart_rank(row_comm, coords, &root);
  MPI_Bcast(local_D, row_size, MPI_DOUBLE, root, row_comm); 
  MPI_Barrier(comm);

  if(column_num==0){
    for(i=0;i<row_size;i++){
      local_x[i]=0;
    }
  }
  MPI_Barrier(comm);


  for(iter=0;iter<max_iter;iter++){
    MPI_Barrier(comm);

    if(column_num==0){
      for (i=0;i<row_size;i++){
	local_P[i]=0;
      }
    }

    MPI_Barrier(comm);



    distributed_matrix_vector_mult(n, local_R, local_x, local_P, comm);

    
    if(column_num==0){
      for (i=0;i<row_size;i++){
	local_x[i]=(local_b[i]-local_P[i])/local_D[i];
      }
    }



    distributed_matrix_vector_mult(n, local_A, local_x, local_w, comm);
    MPI_Barrier(comm);


    coords[0]=0;
    coords[1]=0;
    MPI_Cart_rank(column_comm, coords, &root);

    if(column_num==0){
      l2_norm=0;
      for (i=0;i<row_size;i++){
    	l2_norm=l2_norm+(local_b[i]-local_w[i])*(local_b[i]-local_w[i]);
      }
      l2_norm=sqrt(l2_norm);
      l2=0;
      MPI_Reduce(&l2_norm, &l2, 1, MPI_DOUBLE, MPI_SUM, root, column_comm);
      if(row_num==0){
      }
      if(row_num==0){
      	if(l2<=l2_termination){
      	  signal=1;
      	}
      }
    }
    coords[0]=0;
    coords[1]=0;
    MPI_Cart_rank(comm, coords, &root);
    MPI_Bcast(&signal, 1, MPI_INT, root, comm);
    if(signal==1){
      return;
    }

  }
  return;
}




// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!

  double* local_A = new double[block_decompose_by_dim(n, comm, 0)*block_decompose_by_dim(n, comm, 1)];
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
    //std::cout<<y<<std::endl;
}


// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
  l2_termination=1e-40;
    // distribute the array onto local processors!
  double* local_A = new double[block_decompose_by_dim(n, comm, 0)*block_decompose_by_dim(n, comm, 1)];
    double* local_b = new double[block_decompose_by_dim(n, comm, 0)];

    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);
    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
