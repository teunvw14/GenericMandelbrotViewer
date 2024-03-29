#include <stdio.h>
#include <stdbool.h>
#include <sys/timeb.h>

#include "controls.h"
#include "../application.h"
#include "../cuda_launch.h"
#include "../constants.h"
#include "../global.h"
#include "../mandelbrot_image.h"


// Performance testing functions below:
void start_performance_test(mandelbrot_image** image_ptr, mandelbrot_image* image)
{
    printf("Starting performance test.\n");
    g_start_performance_test_flag = false;
    g_performance_iterations_done = 0;
    g_rendering_done = false;
    reset_render_objects(image);
    ftime(&g_perftest_start);

    // Set parameters for test:
    g_max_iterations_store = image->max_iterations;
    g_resolution_x_store = image->resolution_x;
    g_resolution_y_store = image->resolution_y;
    g_center_real_store = image->center_real;
    g_center_imag_store = image->center_imag;
    g_draw_radius_x_store = image->draw_radius_x;
    g_draw_radius_y_store = image->draw_radius_y;
    g_coloring_mode_store = g_coloring_mode;
    g_coloring_mode = COLORING_SMOOTH;
}

void setup_performance_iteration(mandelbrot_image* image)
{
    int starting_max_iterations = 0;
    // Choose a spot to move to, and change starting max_iterations accordingly
    unsigned short spot = (unsigned short)ceil(4.0 * (float)g_performance_iterations_done / (float)g_performance_iterations_total);
    if (spot <= 0) {
        spot = 1;
    }
    switch (spot) {
    case 1:
        image->center_real = -1.769249938555972345710642912303869;
        image->center_imag = -0.05694208981877081632294590463061;
        image->draw_radius_x = 1.9 * pow(10, -13);
        image->draw_radius_y = image->draw_radius_x;
        starting_max_iterations = 712;
        break;
    case 2:
        image->center_real = -0.0452407411;
        image->center_imag = 0.9868162204352258;
        image->draw_radius_x = 4.4 * pow(10, -9);
        image->draw_radius_y = image->draw_radius_x;
        starting_max_iterations = 328;
        break;
    case 3:
        image->center_real = -0.7336438924199521;
        image->center_imag = 0.2455211406714035;
        image->draw_radius_x = 4.5 * pow(10, -14);
        image->draw_radius_y = image->draw_radius_x;
        starting_max_iterations = 824;
        break;
    case 4:
        image->center_real = -0.0452407411;
        image->center_imag = 0.9868162204352258;
        image->draw_radius_x = 4.4 * pow(10, -9);
        image->draw_radius_y = image->draw_radius_x;
        starting_max_iterations = 328;
        break;
    default:
        break;
    }
    // hack to start iterating from `starting_max_iterations` for each new spot
    image->max_iterations = starting_max_iterations + g_performance_iterations_done * 4 - ((g_performance_iterations_total - 1) * 4 * (spot - 1) / 4);
    printf("\rRendering spot %d with %d iterations.", spot, image->max_iterations);
    fflush(stdout);

    reset_render_objects(image);
}

int end_performance_test(mandelbrot_image* image)
{
    ftime(&g_end);
    image->max_iterations = g_max_iterations_store;
    image->center_real = g_center_real_store;
    image->center_imag = g_center_imag_store;
    image->draw_radius_x = g_draw_radius_x_store;
    image->draw_radius_y = g_draw_radius_y_store;
    g_coloring_mode = g_coloring_mode_store;
    reset_render_objects(image);
    int elapsed_time = (int)1000.0 * (g_end.time - g_perftest_start.time) + (g_end.millitm - g_perftest_start.millitm);
    return elapsed_time;
}
