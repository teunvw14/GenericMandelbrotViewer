#pragma once
#include <stdbool.h>
#include <math.h>

#include "mandelbrot_image.h"
#include "constants.h"
#include "global.h"
#include "util/color_palette.h"

// TODO: update this file with new globals;

void setup_debugging_performance_parameters(void)
{
    g_debugging_enabled = false;
    g_start_performance_test_flag = false;
    g_performance_iterations_done = 0;
    g_performance_iterations_total = 32;
}

void setup_image_parameters(mandelbrot_image* image)
{
    image->center_real = 0.0;
    image->center_imag = 0.0;
    // For some (currently unknown) reason, the resolution components have to be an (identical!) multiple of four.
    image->resolution_x = 512;
    image->resolution_y = 512;
    image->draw_radius_x = 2.5;
    image->draw_radius_y = 2.5;
    image->escape_radius_squared = 4;
    image->max_iterations = 64;
}

void setup_behavioral_parameters(void)
{
    g_application_mode = MODE_VIEW;
    g_coloring_mode = COLORING_SIMPLE;
    g_coloring_palette = palette_pretty;
    g_create_image_flag = false;
    g_rendering_done = false;
    g_lowres_rendering_done = false;
    g_lowres_block_size = 16;
    g_medres_block_size = 9;
}

void setup_cuda_paramaters(mandelbrot_image* image)
{
    g_cuda_block_size = 256;
    g_cuda_num_blocks = (int) ceil(image->resolution_x * image->resolution_y / g_cuda_block_size);
}

void setup_starting_parameters(void)
{
    setup_behavioral_parameters();
    setup_debugging_performance_parameters();
}

void setup_image(mandelbrot_image* image)
{
    setup_image_parameters(image);
    if (g_cuda_device_available) {
        setup_cuda_paramaters(image);
    }
}
