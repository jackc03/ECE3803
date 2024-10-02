#ifndef BLUR_DEVICE_CUH
#define BLUR_DEVICE_CUH

#include <stdio.h>
#include <stdlib.h>


#define GAUSSIAN_SIDE_WIDTH 10
#define GAUSSIAN_SIZE (2 * GAUSSIAN_SIDE_WIDTH + 1)

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


// This function will conduct the convolution for a particular thread index
// given all the other inputs. (See how this function is called in blur.cu
// to get an understanding of what it should do.
__device__ void cuda_blur_kernel_convolution(int thread_index,
                                                float* gpu_raw_data,
                                                float* gpu_blur_v,
                                                float* gpu_out_data,
                                                int n_frames,
                                                int blur_v_size);

// This function will be called from the host code to invoke the kernel
// function. Any memory address/pointer locations passed to this function
// must be host addresses. This function will be in charge of allocating
// GPU memory, invoking the kernel, and cleaning up afterwards. The result
// will be stored in out_data. The function returns the amount of time that
// it took for the function to complete (prior to returning) in milliseconds.
float cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size);

#endif
