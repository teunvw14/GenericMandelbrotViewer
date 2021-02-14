#include <cuda_runtime.h>
#include <stdio.h>
#include "mandelbrot_image.h"
#include "calculations_non_cuda.h"
#include "global.h"

void check_cuda_err()
{
    cudaError_t code =  cudaGetLastError();
    if (code != cudaSuccess)
    {
        char* file = __FILE__;
        int line = __LINE__;
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernels!
extern void launch_build_complex_grid_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_reset_render_arrays_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_mandelbrot_iterate_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_color_cuda(int num_blocks, int block_size, mandelbrot_image* image);

// Build up a grid of complex numbers to iterate
void build_complex_grid(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        launch_build_complex_grid_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        check_cuda_err();
        cudaDeviceSynchronize();
    } else if (!(g_cuda_device_available)) {
        build_complex_grid_non_cuda(image);
    }
}

void mandelbrot_iterate_and_color(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        launch_mandelbrot_iterate_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        check_cuda_err();
        launch_color_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        check_cuda_err();
        cudaDeviceSynchronize();
    } else if (!(g_cuda_device_available)) {
        mandelbrot_iterate_non_cuda(image);
        color_non_cuda(image);
    }
}

// Under maintenance
void mandelbrot_iterate_n_and_color(mandelbrot_image* image, int iterations)
{
    mandelbrot_iterate_and_color(image);
    //mandelbrot_iterate_and_color_cuda << < g_cuda_num_blocks, g_cuda_block_size >> > (iterations, escape_radius_squared, resolution_x, image->resolution_y, points, iterated_points, squared_absolute_values, pixels_rgb);
}

// Reset all the variables that are used for rendering the Mandelbrot
void reset_render_objects(mandelbrot_image* image)
{
    // Reset the `squared_absolute_values` to zero by allocating the memory space again.
    if (g_cuda_device_available) {
        launch_reset_render_arrays_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        check_cuda_err();
        cudaDeviceSynchronize();
    } else if (!(g_cuda_device_available)) {
        reset_render_arrays_non_cuda(image);
    }
    // Rebuild the grid of complex numbers based on (new) center_real and (new) center_imag.
    build_complex_grid(image);

    // Reset the amount of rendered iterations to 0.
    g_rendered_iterations = 0;
}
