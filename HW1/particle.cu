#include <stdlib.h>
#include <stdio.h>

#include "util.h"

#define EPSILON 0.00001f

// amount of floating point numbers between answer and computed value 
// for the answer to be taken correctly
const int maxUlps = 1000;

event_pair timer;
  
float4 host_force_calc(float4 A, float4 B) {
  float x = B.x - A.x;
  float y = B.y - A.y;
  float z = B.z - A.z;
  float rsq = x*x + y*y + z*z;
  // avoid divide by zero
  if(rsq < EPSILON) {
    rsq += EPSILON;
  }
  float r = sqrt(rsq); // distance between particles

  // Coulomb's law
  float f = A.w * B.w / rsq;
  float inv_r = 1.0f / r;

  // force vector has a magnitude of 1 and points along the axis of the charge
  float4 fv = make_float4(x*inv_r, y*inv_r, z*inv_r, f);
  return fv;
}
 
void host_force_eval(float4 *set_A, float4 *set_B, int *indices, float4 *force_vectors, int array_length) {
  for(int i=0; i < array_length; i++) {
    if(indices[i] < array_length && indices[i] >= 0) {
      force_vectors[i] = host_force_calc(set_A[i], set_B[indices[i]]);
    } else {
      force_vectors[i] = make_float4(0.0,0.0,0.0,0.0);
    }
  }
}

__device__ float4 device_calc(float4 A, float4 B) {
  float x = B.x - A.x;
  float y = B.y - A.y;
  float z = B.z - A.z;
  float rsq = x*x + y*y + z*z;

  // avoid divide by zero
  if(rsq < EPSILON) {
    rsq += EPSILON;
  }
  float r = sqrt(rsq); // distance between particles

  // Coulomb's law
  float f = A.w * B.w / rsq;
  float inv_r = 1.0f / r;

  // force vector has a magnitude of 1 and points along the axis of the charge
  float4 fv = make_float4(x*inv_r, y*inv_r, z*inv_r, f);
  return fv;
}


__global__ void force_eval(float4 *set_A, float4 *set_B, int *indices, float4 *force_vectors, int array_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(indices[i] < array_length && indices[i] >= 0) {
    force_vectors[i] = device_calc(set_A[i],  set_B[indices[i]]);
  } else {
    force_vectors[i] = make_float4(0.0,0.0,0.0,0.0);
  }
}


void charged_particles(float4 *h_set_A, float4 *h_set_B, int *h_indices, float4 *h_force_vectors, int num_elements) { 
  // Declare pointers for all the cudaMallocs
  float4 *set_A, *set_B, *force_vectors;
  int *indices;


  // Malloc each pointer
  size_t size = num_elements * sizeof(float4);
  cudaMalloc(&force_vectors, size);
  cudaMalloc(&set_A,         size);
  cudaMalloc(&set_B,         size);
  cudaMalloc(&indices,       num_elements * sizeof(int));


  // Now copy memory over
  cudaMemcpy(force_vectors, h_force_vectors, size, cudaMemcpyHostToDevice);
  cudaMemcpy(set_A, h_set_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(set_B, h_set_B, size, cudaMemcpyHostToDevice);
  cudaMemcpy(indices, h_indices, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  

  // Stealing parameters from part 1
  int block_size = 1024;
  int grid_size = (num_elements + block_size - 1) / block_size;

  
  // Launch kernels
  start_timer(&timer);
  force_eval<<<grid_size, block_size>>>(set_A, set_B, indices, force_vectors,  num_elements);


  check_launch("gpu force eval");
  stop_timer(&timer,"gpu force eval");
  

  // Now copy memory over
  cudaMemcpy(h_force_vectors, force_vectors, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_set_A, set_A, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_set_B, set_B, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_indices, indices, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(force_vectors);
  cudaFree(set_A);
  cudaFree(set_B);
  cudaFree(indices);
}


int main(void) {
  // create arrays of 4M elements
  int num_elements =  1 << 22;

  // pointers to host & device arrays
  float4 *h_set_A = 0;
  float4 *h_set_B = 0;
  int *h_indices = 0;
  float4 *h_force_vectors = 0;
  float4 *h_force_vectors_checker = 0;
  
   // initialize
  srand(time(NULL)); 
  
  // malloc host array
  h_set_A = (float4*)malloc(num_elements * sizeof(float4));
  h_set_B = (float4*)malloc(num_elements * sizeof(float4));
  h_indices = (int*)malloc(num_elements * sizeof(int));
  h_force_vectors = (float4*)malloc(num_elements * sizeof(float4));
  h_force_vectors_checker = (float4*)malloc(num_elements * sizeof(float4));
  
  // if either memory allocation failed, report an error message
  if(h_set_A == 0 || h_set_B == 0 || h_force_vectors == 0 || h_indices == 0 || h_force_vectors_checker == 0) {
    printf("couldn't allocate memory\n");
    exit(1);
  }

  // generate random input
  for(int i=0; i < num_elements; i++) {
    h_set_A[i] = make_float4(rand(), rand(), rand(), rand()); 
    h_set_B[i] = make_float4(rand(), rand(), rand(), rand());

    // some indices will be invalid
    h_indices[i] = rand() % (num_elements + 2);
  }
  
  start_timer(&timer);
  // generate reference output
  host_force_eval(h_set_A, h_set_B, h_indices, h_force_vectors_checker, num_elements);
  check_launch("host force eval");
  stop_timer(&timer, "host force eval");
  
  // the results of the calculation need to end up in h_force_vectors;
  charged_particles(h_set_A, h_set_B, h_indices, h_force_vectors, num_elements);
  
  // check CUDA output versus reference output
  int error = 0;
  
  for(int i=0; i<num_elements; i++) {
    float4 v = h_force_vectors[i];
    float4 vc = h_force_vectors_checker[i];

    if( !AlmostEqual2sComplement(v.x,vc.x,maxUlps) ||
      !AlmostEqual2sComplement(v.y,vc.y,maxUlps) ||
      !AlmostEqual2sComplement(v.z,vc.z,maxUlps) ||
      !AlmostEqual2sComplement(v.w,vc.w,maxUlps)) { 
      error = 1;
    }
  }
  
  if(error) {
    printf("Failure: Output of CUDA version and CPU version didn't match\n");
  }
  else {
    printf("Success: CUDA and reference output match\n");
  }
 
 
  // deallocate memory
  free(h_set_A);
  free(h_set_B);
  free(h_indices);
  free(h_force_vectors);
  free(h_force_vectors_checker);
}

