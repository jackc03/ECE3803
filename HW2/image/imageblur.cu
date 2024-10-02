#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "util.h"
#include "ppm.h"

#define IMAGE_DIM 2048
#define SAMPLE_SIZE 10
#define SAMPLE_DIM (SAMPLE_SIZE*2+1)
#define NUMBER_OF_SAMPLES (SAMPLE_DIM*SAMPLE_DIM)


/************************************/
/* Kernel for blur using GPU memory */
/************************************/
__global__ void image_blur(uchar4 *image, uchar4 *image_output) {
    // TODO calculate array index based on special CUDA variables
    // TODO loop through adjacent pixel coordinates
    // TODO read pixels from row-major array
    // TODO calculate average of surrounding pixels
    // TODO store averaged pixel in image_output

    // get the x and y index along with grid width
    int index_x = blockIdx.x * blockDim.x + threadIdx.x;    
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;

    // use previous variables to get the global index 
    int thread_index = index_y * grid_width + index_x;

    // variable that keeps current x, y and combined indexes
    int curr_combined, curr_x, curr_y;
    curr_combined = thread_index;


    // int which_row = 1;
    // if (thread_index > 2048 * which_row  + 150 && thread_index < 2048 * (which_row) + 200 ) {
    //     printf
    // }
    
    // start at top left of sample square
    curr_x = index_x;
    curr_y = index_y;
    for (int i = 0; i < SAMPLE_DIM; ++i) {
        for (int j = 0; j < SAMPLE_DIM; ++j) {

        }

    }

    
}


/************************************/
/* Kernel for blur using 1D texture */
/************************************/
__global__ void image_blur_texture1D(cudaTextureObject_t sample1D, uchar4 *image_output) {
    // TODO calculate array index based on special CUDA variables
    // TODO loop through adjacent pixel coordinates
    // TODO read pixels from texture sample1D
    //        use syntax: uchar4 = tex1Dfetch<uchar4>(sample1D, offset)
    // TODO calculate average of surrounding pixels
    // TODO store averaged pixel in image_output
}


/************************************/
/* Kernel for blur using 2D texture */
/************************************/
__global__ void image_blur_texture2D(cudaTextureObject_t sample2D, uchar4 *image_output) {
    // TODO loop through adjacent pixel coordinates
    // TODO read pixels from texture sample1D
    //        use syntax: uchar4 pixel = tex2D<uchar4>(sample2D, (float)x_offset, (float)y_offset)
    // TODO calculate average of surrounding pixels
    // TODO store averaged pixel in image_output
}


int main(int argc, char **argv) {
    unsigned int image_size;
    uchar4 *d_image, *d_image_output;
    uchar4 *h_image;
    cudaEvent_t start, stop;

    image_size = IMAGE_DIM * IMAGE_DIM * sizeof(uchar4);

    if (argc != 3) {
        printf("Syntax: %s mode outputfilename.ppm\n\twhere mode is 0, 1, or 2\n", argv[0]);
        exit(1);
    }
    int mode = atoi(argv[1]);
    const char *filename = argv[2];

    // create timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory on the GPU for the output image
    CHECK_ERROR(cudaMalloc((void**)&d_image, image_size));
    CHECK_ERROR(cudaMalloc((void**)&d_image_output, image_size));

    // allocate and load host image
    h_image = (uchar4*)malloc(image_size);
    if (h_image == NULL) {
        printf("Malloc failed");
        exit(1);
    }
    input_image_file("input.ppm", h_image, IMAGE_DIM);

    // copy image to device memory
    CHECK_ERROR(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice));

    //cuda layout and execution
    dim3    blocksPerGrid(IMAGE_DIM / 16, IMAGE_DIM / 16);
    dim3    threadsPerBlock(16, 16);

    switch (mode) {


        /*************************/
        /* Blur using GPU memory */
        /*************************/
        case 0:
        {
            // normal version
            cudaEventRecord(start, 0);
            image_blur<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_image_output);
            check_launch("kernel normal");
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            printf("Blur using device memory, time: %f\n", ms);
        }
        break;


        /*************************/
        /* Blur using 1D texture */
        /*************************/
        case 1:
        {
            cudaTextureObject_t sample1d=0;

            /*
                We use this cudaResourceDesc to describe the
                structure of our data so CUDA can do indexing for us.
                The relevant details: linear array, of unsigned values,
                such that every element consists of four 8-bit values.
            */
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypeLinear;
            resDesc.res.linear.devPtr = d_image;

            resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
            resDesc.res.linear.desc.x = resDesc.res.linear.desc.y = 
                resDesc.res.linear.desc.z = resDesc.res.linear.desc.w = 8; // bits per channel
            resDesc.res.linear.sizeInBytes = image_size;

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;

            CHECK_ERROR(cudaCreateTextureObject(&sample1d, &resDesc, &texDesc, NULL));

            cudaEventRecord(start, 0);
            image_blur_texture1D<<<blocksPerGrid, threadsPerBlock>>>(sample1d, d_image_output);
            check_launch("kernel tex1D");
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cudaDestroyTextureObject(sample1d);
            printf("Blur using 1D texture, time: %f\n", ms);
        }
        break;

        /*************************/
        /* Blur using 2D texture */
        /*************************/
        case 2:
        {
            cudaTextureObject_t sample2d=0;

            /*
                We use this cudaResourceDesc to describe the
                structure of our data so CUDA can do indexing for us.
                The relevant details: pitched-2D array with no padding
                between rows, of unsigned values, where each array element
                consists of four 8-bit values, and there are a total of
                IMAGE_DIM * IMAGE_DIM elements.
            */
            cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(resDesc));
            resDesc.resType = cudaResourceTypePitch2D;

            resDesc.res.pitch2D.devPtr = d_image;
            resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
            resDesc.res.pitch2D.desc.x = resDesc.res.linear.desc.y = 
                resDesc.res.linear.desc.z = resDesc.res.linear.desc.w = 8; // bits per channel
            resDesc.res.pitch2D.width = IMAGE_DIM;
            resDesc.res.pitch2D.height = IMAGE_DIM;
            resDesc.res.pitch2D.pitchInBytes = IMAGE_DIM * sizeof(uchar4);

            cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(texDesc));
            texDesc.readMode = cudaReadModeElementType;

            CHECK_ERROR(cudaCreateTextureObject(&sample2d, &resDesc, &texDesc, NULL));

            cudaEventRecord(start, 0);
            image_blur_texture2D<<<blocksPerGrid, threadsPerBlock>>>(sample2d, d_image_output);
            check_launch("kernel tex2D");
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cudaDestroyTextureObject(sample2d);
            printf("Blur using 2D texture, time: %f\n", ms);
        }
        break;

        default:
            printf("Unknown mode %d\n", mode);
            exit(1);
            break;
    }


    // copy the image back from the GPU for output to file
    CHECK_ERROR(cudaMemcpy(h_image, d_image_output, image_size, cudaMemcpyDeviceToHost));

    // output image
    output_image_file(filename, h_image, IMAGE_DIM);

    //cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_image);
    cudaFree(d_image_output);
    free(h_image);

    return 0;
}
