#pragma once
#include "../mandelbrot_image.h"

// Performance testing functions below:
void start_performance_test(mandelbrot_image** image_ptr, mandelbrot_image* image);
void setup_performance_iteration(mandelbrot_image* image);
int end_performance_test(mandelbrot_image* image);