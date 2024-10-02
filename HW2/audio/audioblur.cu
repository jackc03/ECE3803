/* 
 * CUDA audio blur
 */

#include "audioblur.cuh"
#include <iostream>

#include <cstdio>
#include <cuda_runtime.h>

// for my reference while coding
// //                bottom 10
// float kernel[] = {0.0111947, 0.0163699,0.0229988, 0.0310452, 0.0402634,
//                   0.0501713, 0.0600659, 0.0690923, 0.0763588, 0.0810805, 
                  
//                   //middle val
//                   0.0827185, 

//                   // top 10
//                   0.0810805, 0.0763588, 0.0690923, 0.0600659, 0.0501713, 
//                   0.0402634, 0.0310452, 0.0229988,0.0163699, 0.0111947};


__device__ void cuda_blur_kernel_convolution(uint thread_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
    // TODO: Implement the necessary convolution function that should be
    //       completed for each thread_index. Use the CPU implementation in
    //       blur.cpp as a reference.
    
    // get thread id;
    int i,j,k;
    i = thread_index;
    j = i - 1;
    k = i + 1;

    if (i < n_frames){
        gpu_out_data[i] = gpu_raw_data[i] * gpu_blur_v[10];


        // Start j at i - 1 and decrement until u get to 0 or finish 10 increments
        // Start k at i + 1 and increment until u get to n_frames or finish 10 increments
        while ( (j >= (i - (blur_v_size / 2))) || (k <= (i + (blur_v_size / 2))) ) {
            // check boundary conditions before adding
            if (j >= 0 ){
                gpu_out_data[i] += gpu_raw_data[j] * gpu_blur_v[j - i + 10];
            } else {
                j = i - (blur_v_size / 2) - 1;
            }
            if (k < n_frames) {
                gpu_out_data[i] += gpu_raw_data[k] * gpu_blur_v[k - i + 10];
                k ++;
            } else {
                k = i + (blur_v_size / 2) + 1;
            }

            // decremenet and increment regardless
            j--;
            k++;

        }

        // for (int i = 0; i < GAUSSIAN_SIZE; i++) {
        //     for (int j = 0; j <= i; j++)
        //         gpu_out_data[i] += gpu_raw_data[i - j] * gpu_blur_v[j]; 
        // }
        // for (int i = GAUSSIAN_SIZE; i < n_frames; i++) {
        //     for (int j = 0; j < GAUSSIAN_SIZE; j++)
        //         gpu_out_data[i] += gpu_raw_data[i - j] * gpu_blur_v[j]; 
        // }
    }

}

__global__ void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    // TODO: Compute the current thread index.
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    int num_convolutions;
    //find how many indeces each thread is responsible for
    num_convolutions = (n_frames + (n_frames - 1)) / (gridDim.x * blockDim.x);
    printf("num convolutions = %d\n", num_convolutions);

    // printf("%d\n", num_convolutions);

    thread_index = thread_index * num_convolutions;


    // TODO: Update the while loop to handle all indices for this thread.
    //       Remember to advance the index as necessary.
    for (int i = thread_index; i < thread_index + num_convolutions; ++i) {
        // Do computation for this thread index
        cuda_blur_kernel_convolution(i, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        printf("%d\n", i);

    }
}

float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    // Use the CUDA machinery for recording time
    cudaEvent_t start_gpu, stop_gpu;
    float time_milli = -1;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    cudaEventRecord(start_gpu);

    // TODO: Allocate GPU memory for the raw input data (either audio file
    //       data or randomly generated data). The data is of type float and
    //       has n_frames elements. Then copy the data in raw_data into the
    //       GPU memory you allocated.

    float* gpu_raw_data;
    HANDLE_ERROR(cudaMalloc((void**)&gpu_raw_data, sizeof(float) * n_frames));
    HANDLE_ERROR(cudaMemcpy(gpu_raw_data, raw_data, sizeof(float) * n_frames, cudaMemcpyHostToDevice));
    


    // TODO: Allocate GPU memory for the impulse signal. The data is of type
    //       float and has blur_v_size elements. Then copy the data in blur_v
    //       into the GPU memory you allocated.

    float* gpu_blur_v;
    HANDLE_ERROR(cudaMalloc((void**)&gpu_blur_v, sizeof(float) * blur_v_size));
    HANDLE_ERROR(cudaMemcpy(gpu_blur_v, blur_v, sizeof(float) * blur_v_size, cudaMemcpyHostToDevice));
  



    // TODO: Allocate GPU memory to store the output audio signal after the
    //       convolution. The data is of type float and has n_frames elements.
    //       Initialize the data as necessary.
    float* gpu_out_data;
    HANDLE_ERROR(cudaMalloc((void**)&gpu_out_data, sizeof(float) * n_frames));



  
    // TODO: Appropriately call the kernel function, specifying
    //       block size and thereads per block
    cuda_blur_kernel<<<blocks, threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);


    // Check for errors on kernel call
    cudaDeviceSynchronize();
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
        exit(1);
    }



    // TODO: Now that kernel calls have finished, copy the output signal
    //       back from the GPU to host memory. (We store this channel's result
    //       in out_data on the host.)
    HANDLE_ERROR(cudaMemcpy(out_data, gpu_out_data, sizeof(float) * n_frames, cudaMemcpyDeviceToHost));



    // TODO: Now that we have finished our computations on the GPU, free the
    //       GPU resources.
    HANDLE_ERROR(cudaFree(gpu_out_data));
    HANDLE_ERROR(cudaFree(gpu_raw_data));
    HANDLE_ERROR(cudaFree(gpu_blur_v));

    

    // Stop the recording timer and return the computation time
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&time_milli, start_gpu, stop_gpu);
    return time_milli;
}
