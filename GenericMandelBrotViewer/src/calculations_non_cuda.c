#include <cuComplex.h>
#include <math.h>
#include "mandelbrot_image.h"
#include "constants.h"
#include "global.h"
#include "util/color_palette.h"
#include "util/hsv_to_rgb.h"


void build_complex_grid_non_cuda(mandelbrot_image* image)
{
    // Create a grid of complex numbers around the center point (center_real, center_imag).

    double step_x = 2 * image->draw_radius_x / image->resolution_x;
    double step_y = 2 * image->draw_radius_y / image->resolution_y;
    double point_re;
    double point_im;
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        point_im = image->center_imag + pixel_y * step_y - image->draw_radius_y;
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            index = pixel_y * image->resolution_x + pixel_x;
            point_re = image->center_real + pixel_x * step_x - image->draw_radius_x;
            image->points[index] = make_cuDoubleComplex(point_re, point_im);
            //image->iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
        }
    }
}

void reset_render_arrays_non_cuda(mandelbrot_image* image)
{
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            index = pixel_y * image->resolution_x + pixel_x;
            image->iterationsArr[index] = 0;
            image->squared_absolute_values[index] = 0;
        }
    }
}

void mandelbrot_iterate_non_cuda(mandelbrot_image* image)
{
    // For every complex point in the current view, calculate the 
    // iterations required for a given point to exceed the escape radius.

    int index;
    unsigned int iterations_;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y += 1) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x += 1) {
            index = pixel_y * image->resolution_x + pixel_x;
            // `cnum` is the complex number that we'll iterate
            cuDoubleComplex cnum = (image->points)[index];
            double re = cnum.x;
            double im = cnum.y;
            // The below variable exists so that we can change `re`
            // without it affecting the calculation for the new
            // value of `im`
            double re_temp;
            double sq_abs = (image->squared_absolute_values)[index];
            iterations_ = 0;
            while (iterations_ < image->max_iterations && sq_abs < image->escape_radius_squared) {
                re_temp = re * re - im * im + cnum.x;
                im = 2 * re * im + cnum.y;
                re = re_temp;
                sq_abs = re * re + im * im;
                iterations_++;
            }
            (image->iterationsArr)[index] = iterations_;
            (image->squared_absolute_values)[index] = sq_abs;
        }
    }
}

void mandelbrot_iterate_non_cuda_downscaled(mandelbrot_image* image, unsigned int downscale_factor) {
    // Do the same as above, but use the iterations calculated for one pixel
    // for a full block of pixels

    int index;
    unsigned int iterations_;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y += downscale_factor) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x += downscale_factor) {
            index = pixel_y * image->resolution_x + pixel_x;
            // `cnum` is the complex number that we'll iterate
            cuDoubleComplex cnum = (image->points)[index];
            double re = cnum.x;
            double im = cnum.y;
            // The below variable exists so that we can change `re`
            // without it affecting the calculation for the new
            // value of `im`
            double re_temp;
            double sq_abs = (image->squared_absolute_values)[index];
            iterations_ = 0;
            while (iterations_ < image->max_iterations && sq_abs < image->escape_radius_squared) {
                re_temp = re * re - im * im + cnum.x;
                im = 2 * re * im + cnum.y;
                re = re_temp;
                sq_abs = re * re + im * im;
                iterations_++;
            }
            for (int block_y = pixel_y; block_y < pixel_y + downscale_factor && block_y < image->resolution_y; block_y++) {
                for (int block_x = pixel_x; block_x < pixel_x + downscale_factor && block_x < image->resolution_x; block_x++) {
                    index = block_y * image->resolution_x + block_x;
                    (image->iterationsArr)[index] = iterations_;
                    (image->squared_absolute_values)[index] = sq_abs;
                }
            }
        }
    }
}


void color_simple_non_cuda(mandelbrot_image* image) {
    int index;
    unsigned int iterations;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_x + pixel_x;
            iterations = (image->iterationsArr)[index];
            color_rgb pixel_color = white;
            if (iterations >= image->max_iterations) {
                pixel_color = black;
            }
            // Set the RGB values in the array
            (image->pixels_rgb)[3 * index + 0] = pixel_color.r; // Red value
            (image->pixels_rgb)[3 * index + 1] = pixel_color.g; // Green value
            (image->pixels_rgb)[3 * index + 2] = pixel_color.b; // Blue value
        }
    }
}

void color_palette_non_cuda(mandelbrot_image* image, palette plt) {
    int index;
    unsigned int iterations;
    color_rgb pixel_color;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_x + pixel_x;
            iterations = (image->iterationsArr)[index];
            pixel_color = black;
            if (iterations < image->max_iterations) {
                palette p = palette_pretty;
                int color_index = iterations % p.length;
                // smooth color to make it a little easier on the eyes
                int next_color_index = (color_index + 1) % p.length;
                float escape_size = (float)(image->squared_absolute_values[index]);
                float lerp_factor = 1 - log2f(log(escape_size));
                pixel_color = lerp_color(p.colors[color_index], p.colors[next_color_index], lerp_factor);
            }
            // Set the RGB values in the array
            (image->pixels_rgb)[3 * index + 0] = pixel_color.r; // Red value
            (image->pixels_rgb)[3 * index + 1] = pixel_color.g; // Green value
            (image->pixels_rgb)[3 * index + 2] = pixel_color.b; // Blue value
        }
    }
}

void color_smooth_non_cuda(mandelbrot_image* image)
{
    // Do some smooth coloring!
    int index;
    unsigned int iterations;
    color_rgb pixel_color;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_x + pixel_x;
            iterations = (image->iterationsArr)[index];
            pixel_color = black;
            if (iterations < image->max_iterations) {
                float f_iterations = (float)iterations;
                float f_max_iterations = (float)image->max_iterations;
                // Smooth colors!
                float escape_size = (float)(image->squared_absolute_values[index]);
                float smoothed_iterations = iterations + 1 - log2f(log(escape_size));
                float H = 360 * smoothed_iterations / f_max_iterations;
                float S = 0.7f;
                float V = 1.0f;

                if (H > 360) H = 360;
                if (H < 0) H = 0;
                pixel_color = hsv_to_rgb(H, S, V);
            }
            (image->pixels_rgb)[3 * index + 0] = pixel_color.r; // Red value
            (image->pixels_rgb)[3 * index + 1] = pixel_color.g; // Green value
            (image->pixels_rgb)[3 * index + 2] = pixel_color.b; // Blue value
        }
    }
}

void color_non_cuda(mandelbrot_image* image, palette plt) {
    switch (g_coloring_mode) {
    case COLORING_SIMPLE:
        color_simple_non_cuda(image);
        break;
    case COLORING_PALETTE:
        color_palette_non_cuda(image, plt);
        break;
    case COLORING_SMOOTH:
        color_smooth_non_cuda(image);
        break;
    }
}
