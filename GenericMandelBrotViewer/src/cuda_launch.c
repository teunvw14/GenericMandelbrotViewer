#include <stdio.h>

#include <cuda_runtime.h>
#include "mandelbrot_image.h"
#include "calculations_non_cuda.h"
#include "constants.h"
#include "global.h"

void check_cuda_err(void)
{
    cudaError_t code =  cudaGetLastError();
    if (code != cudaSuccess)
    {
        char* file = __FILE__;
        int line = __LINE__;
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}

// Kernels!
extern void launch_build_complex_grid_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_reset_render_arrays_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_mandelbrot_iterate_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_color_smooth_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_color_palette_cuda(int num_blocks, int block_size, mandelbrot_image* image, palette plt);

// Build up a grid of complex numbers to iterate
void build_complex_grid(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        launch_build_complex_grid_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        cudaDeviceSynchronize();
    } else if (!(g_cuda_device_available)) {
        build_complex_grid_non_cuda(image);
    }
}

void color_cuda(mandelbrot_image* image, palette plt) {
    launch_color_smooth_cuda(g_cuda_num_blocks, g_cuda_block_size, image, plt);
    // TODO: add more CUDA coloring modes
    //switch (g_coloring_mode) {
    //case COLORING_PALETTE:
    //    launch_color_palette_cuda(g_cuda_num_blocks, g_cuda_block_size, image, plt);
    //    break;
    //case COLORING_SMOOTH:
    //    launch_color_smooth_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
    //    break;
    //default:
    //    launch_color_smooth_cuda(g_cuda_num_blocks, g_cuda_block_size, image, plt);
    //    break;
    //}
}

void mandelbrot_color(mandelbrot_image* image) {
    if (g_cuda_device_available) {
        color_cuda(image, g_coloring_palette);
    }
    else {
        color_non_cuda(image, g_coloring_palette);
    }
}

void mandelbrot_iterate(mandelbrot_image* image) {
    if (g_cuda_device_available) {
        launch_mandelbrot_iterate_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
    }
    else {
        mandelbrot_iterate_non_cuda(image);
    }
}

void mandelbrot_iterate_n(mandelbrot_image* image, int n) {
    if (g_cuda_device_available) {
        launch_mandelbrot_iterate_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
    }
    else {
        mandelbrot_iterate_non_cuda(image);
    }
}

void mandelbrot_iterate_and_color(mandelbrot_image* image)
{
    mandelbrot_iterate(image);
    mandelbrot_color(image);
}

// Under maintenance
void mandelbrot_iterate_n_and_color(mandelbrot_image* image, int n)
{
    mandelbrot_iterate_n(image, n);
    mandelbrot_color(image);
    if (g_cuda_device_available) {
        cudaDeviceSynchronize();
    }
}

// Reset all the variables that are used for rendering the Mandelbrot
void reset_render_objects(mandelbrot_image* image)
{
    // Reset the `squared_absolute_values` to zero by allocating the memory space again.
    if (g_cuda_device_available) {
        launch_reset_render_arrays_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
        cudaDeviceSynchronize();
    } else if (!(g_cuda_device_available)) {
        reset_render_arrays_non_cuda(image);
    }
    // Rebuild the grid of complex numbers based on (new) center_real and (new) center_imag.
    build_complex_grid(image);

    // Reset the amount of rendered iterations to 0.
    g_rendered_iterations = 0;
}
