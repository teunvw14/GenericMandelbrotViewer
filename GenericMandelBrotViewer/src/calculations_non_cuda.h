#pragma once
#include "mandelbrot_image.h"

void build_complex_grid_non_cuda(mandelbrot_image* image);
void reset_render_arrays_non_cuda(mandelbrot_image* image);
void mandelbrot_iterate_non_cuda(mandelbrot_image* image);
void color_non_cuda(mandelbrot_image* image);