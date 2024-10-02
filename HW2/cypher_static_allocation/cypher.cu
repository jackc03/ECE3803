#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include "util.h"


#define NUM_ELEMENTS 16777216

__device__ unsigned int device_input_array[NUM_ELEMENTS];
__device__ unsigned int device_output_array[NUM_ELEMENTS];


void host_shift_cypher(unsigned int *input_array, unsigned int *output_array, unsigned int shift_amount, unsigned int alphabet_max, unsigned int array_length) {
  for(unsigned int i=0; i < array_length; i++) {
    int element = input_array[i];
    int shifted = element + shift_amount;
    if(shifted > alphabet_max) {
      shifted = shifted % (alphabet_max + 1);
    }
    output_array[i] = shifted;
  }
}


// This kernel implements a per element shift
__global__ void shift_cypher(unsigned int shift_amount, unsigned int alphabet_max, unsigned int array_length) {
  // Find actual shift index
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Store result in output array
  device_output_array[i] = (device_input_array[i] + shift_amount) % (alphabet_max + 1);
}


int main(void) {
  // initialize
  srand(time(NULL));
  

  
  unsigned int alphabet_max = 45647;
  
  // compute the size of the arrays in bytes
  int num_bytes = NUM_ELEMENTS * sizeof(unsigned int);

  // pointers to host & device arrays
  unsigned int *host_input_array = 0;
  unsigned int *host_output_array = 0;
  unsigned int *host_output_checker_array = 0;
  // unsigned int *device_input_array = 0;
  // unsigned int *device_output_array = 0;
  
  event_pair timer;
  

  // malloc host arrays
  host_input_array = (unsigned int*)malloc(num_bytes);
  host_output_array = (unsigned int*)malloc(num_bytes);
  host_output_checker_array = (unsigned int*)malloc(num_bytes);

  // if either memory allocation failed, report an error message
  if(host_input_array == 0 || host_output_array == 0 || host_output_checker_array == 0) {
    printf("couldn't allocate memory\n");
    return 1;
  }

  // cudaMalloc device arrays
  // CHECK_ERROR(cudaMalloc((void**)&device_input_array, num_bytes));
  // CHECK_ERROR(cudaMalloc((void**)&device_output_array, num_bytes));

  
  
  // generate random input string
  unsigned int shift_amount = rand();
  
  for(int i=0;i < NUM_ELEMENTS;i++) {
    host_input_array[i] = (unsigned int)rand(); 
  }
  
  // do copies to and from gpu once to get rid of timing weirdness
  CHECK_ERROR(cudaMemcpyToSymbol(device_input_array, host_input_array, num_bytes, 0, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpyFromSymbol(host_output_array, device_output_array, num_bytes, 0, cudaMemcpyDeviceToHost));


  start_timer(&timer);
  // copy input to GPU
  CHECK_ERROR(cudaMemcpyToSymbol(device_input_array, host_input_array, num_bytes, 0, cudaMemcpyHostToDevice));
  stop_timer(&timer,"copy to gpu");
  
  // choose a number of threads per block
  // we use 512 threads here
  int block_size = 512;

  int grid_size = (NUM_ELEMENTS + block_size - 1) / block_size;

  start_timer(&timer);
  // launch kernel
  shift_cypher<<<grid_size,block_size>>>(shift_amount, alphabet_max, NUM_ELEMENTS);
  check_launch("gpu shift cypher");
  stop_timer(&timer,"gpu shift cypher");

  start_timer(&timer);
  // download and inspect the result on the host:
  CHECK_ERROR(cudaMemcpyFromSymbol(host_output_array, device_output_array, num_bytes, 0, cudaMemcpyDeviceToHost));
  stop_timer(&timer,"copy from gpu");
  
  start_timer(&timer);
  // generate reference output on CPU
  host_shift_cypher(host_input_array, host_output_checker_array, shift_amount, alphabet_max, NUM_ELEMENTS);
  stop_timer(&timer,"host shift cypher");
  
  // check CUDA output versus reference output
  int error = 0;
  for(int i=0;i<NUM_ELEMENTS;i++) {
    // printf("Host[%d]: %d; GPU[%d]: %d\n", i, host_output_array[i], i, host_output_checker_array[i]);

    if(host_output_array[i] != host_output_checker_array[i]) 
    { 
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
  free(host_input_array);
  free(host_output_array);
  free(host_output_checker_array);
}

