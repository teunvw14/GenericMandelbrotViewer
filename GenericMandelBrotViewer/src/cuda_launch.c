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
extern void launch_mandelbrot_iterate_downscaled_cuda(int num_blocks, int block_size, mandelbrot_image* image, unsigned int downscale_factor);
extern void launch_color_smooth_cuda(int num_blocks, int block_size, mandelbrot_image* image);
extern void launch_color_palette_cuda(int num_blocks, int block_size, mandelbrot_image* image, palette plt);


// Build up a grid of complex numbers to iterate
void build_complex_grid(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        launch_build_complex_grid_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
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

void mandelbrot_iterate_downscaled(mandelbrot_image* image, unsigned int downscale_factor) {
    if (g_cuda_device_available) {
        launch_mandelbrot_iterate_downscaled_cuda(g_cuda_num_blocks, g_cuda_block_size, image, downscale_factor);
    }
    else {
        mandelbrot_iterate_non_cuda_downscaled(image, downscale_factor);
    }
}

void mandelbrot_iterate_and_color(mandelbrot_image* image)
{
    mandelbrot_iterate(image);
    mandelbrot_color(image);
    if (g_cuda_device_available) {
        cudaDeviceSynchronize();
    }
}

void mandelbrot_iterate_downscaled_and_color(mandelbrot_image* image, unsigned int downscale_factor)
{
    mandelbrot_iterate_downscaled(image, downscale_factor);
    mandelbrot_color(image);
    if (g_cuda_device_available) {
        cudaDeviceSynchronize();
    }
}

void reset_render_arrays(mandelbrot_image* image) {
    if (g_cuda_device_available) {
        launch_reset_render_arrays_cuda(g_cuda_num_blocks, g_cuda_block_size, image);
    }
    else if (!(g_cuda_device_available)) {
        reset_render_arrays_non_cuda(image);
    }
}

void reset_render_objects(mandelbrot_image* image)
{
    // Reset all the variables that are used for rendering the Mandelbrot
    reset_render_arrays(image);
    // Rebuild the grid of complex numbers based on (new) center_real and (new) center_imag.
    build_complex_grid(image);
    if (g_cuda_device_available) {
        cudaDeviceSynchronize();
    }

    // Reset the amount of rendered iterations to 0.
    g_rendering_done = false;
    g_lowres_rendering_done = false;
    g_medres_rendering_done = false;
}
