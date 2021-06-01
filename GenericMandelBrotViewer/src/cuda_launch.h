#pragma once
#include "mandelbrot_image.h"

void build_complex_grid(mandelbrot_image* image);
void mandelbrot_iterate_and_color(mandelbrot_image* image);
void mandelbrot_iterate_n_and_color(mandelbrot_image* image, int iterations);
void reset_render_objects(mandelbrot_image* image);
