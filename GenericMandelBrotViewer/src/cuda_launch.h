#pragma once
#include "mandelbrot_image.h"

void reset_render_arrays(mandelbrot_image* image);
void build_complex_grid(mandelbrot_image* image);
void mandelbrot_iterate(mandelbrot_image* image);
void mandelbrot_iterate_and_color(mandelbrot_image* image);
void mandelbrot_iterate_downscaled(mandelbrot_image* image, unsigned int block_size);
void mandelbrot_iterate_downscaled_and_color(mandelbrot_image* image, unsigned int block_size);
void reset_render_objects(mandelbrot_image* image);
