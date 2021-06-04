#pragma once

#include <cuComplex.h>

typedef struct mandelbrot_image {
    int resolution_x;
    int resolution_y;
    int max_iterations;
    double center_real;
    double center_imag;
    double draw_radius_x;
    double draw_radius_y;
    double escape_radius_squared;
    cuDoubleComplex* points;
    //cuDoubleComplex** points_ptr;
    //cuDoubleComplex* iterated_points;
    //cuDoubleComplex** iterated_points_ptr;
    double* squared_absolute_values; // used for smooth coloring
    //double** squared_absolute_values_ptr;
    unsigned char* pixels_rgb;
    //unsigned char** pixels_rgb_ptr;
    unsigned int* iterationsArr; // maybe remove this, might be completely unneeded
    //unsigned int** iterationsArr_ptr;
} mandelbrot_image;
