#pragma once

#include <cuComplex.h>

typedef struct mandelbrot_image {
	double center_real;
	double center_imag;
	int resolution_x;
	int resolution_y;
	double draw_radius;
	double escape_radius_squared;
	int max_iterations;
	cuDoubleComplex* points;
	cuDoubleComplex* iterated_points;
	double* squared_absolute_values;
	unsigned char* pixels_rgb;
	unsigned int* iterationsArr;
} mandelbrot_image;
