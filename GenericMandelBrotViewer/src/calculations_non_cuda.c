#include <cuComplex.h>
#include <math.h>
#include "mandelbrot_image.h"
#include "constants.h"
#include "global.h"
#include "util/color_palette.h"


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
            image->iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
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
    int index = 0;
    int iterations_ = 0;
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_x + pixel_x;
            cuDoubleComplex starting_number = image->points[index];

            cuDoubleComplex iterated_point = image->iterated_points[index];
            double sq_abs = image->squared_absolute_values[index];
            iterations_ = image->iterationsArr[index];
            while (iterations_ < image->max_iterations && sq_abs < image->escape_radius_squared) {
                iterated_point = make_cuDoubleComplex(iterated_point.x * iterated_point.x - iterated_point.y * iterated_point.y + starting_number.x,
                                                      2 * iterated_point.x * iterated_point.y + starting_number.y);
                sq_abs = iterated_point.x * iterated_point.x + iterated_point.y * iterated_point.y;
                iterations_++;
            }
            image->iterated_points[index] = iterated_point;
            image->iterationsArr[index] = iterations_;
            image->squared_absolute_values[index] = sq_abs;
        }
    }
}


void color_non_cuda(mandelbrot_image* image)
{
    // Do some coloring!
    int index;
    unsigned int iterations;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++) {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++) {
            // Calculate the iterations required for a given point to exceed the escape radius.
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_x + pixel_x;
            iterations = (image->iterationsArr)[index];
            if (iterations == image->max_iterations) {
                // Values that don't escape are colored black:
                (image->pixels_rgb)[3 * index + 0] = 0; // Red value
                (image->pixels_rgb)[3 * index + 1] = 0; // Green value
                (image->pixels_rgb)[3 * index + 2] = 0; // Blue value
            }
            else {
                color_rgb pixel_color;
                if (g_coloring_mode == COLORING_SIMPLE) {
                    simple_palette hacker_green_palette;
                    hacker_green_palette.start_color = black;
                    hacker_green_palette.end_color = white;
                    float factor = sqrtf((float)iterations / (float)image->max_iterations);
                    pixel_color = lerp_color(hacker_green_palette.start_color, hacker_green_palette.end_color, factor);
                }
                else if (g_coloring_mode == COLORING_PALETTE) {
                    palette p = palette_pretty;
                    int color_index = iterations % p.length;
                    int next_color_index = (color_index + 1) % p.length;
                    float escape_size = (float)(image->squared_absolute_values[index]);
                    float lerp_factor = 1 - log2f(log(escape_size));
                    if (lerp_factor > 1) lerp_factor = 1;
                    if (lerp_factor < 0) lerp_factor = 0;
                    pixel_color = lerp_color(p.colors[color_index], p.colors[next_color_index], lerp_factor);
                }
                else if (g_coloring_mode == COLORING_SMOOTH) {
                    float f_iterations = (float)iterations;
                    float f_max_iterations = (float)image->max_iterations;
                    // Smooth colors!
                    float escape_size = (float)(image->squared_absolute_values[index]);
                    float smoothed_iterations = iterations + 1 - log2f(log(escape_size));
                    float H = 360 * smoothed_iterations / f_max_iterations;
                    float S = .65;
                    float V = 1;

                    if (H > 360) H = 360;
                    if (H < 0) H = 0;
                    // HSV to RGB conversion, yay!
                    // TODO: look into edge cases for H and why they happen.
                    //if (H > 360 || H < 0 || S > 1 || S < 0 || V > 1 || V < 0)
                    //{
                    //printf("The given HSV values are not in valid range.\n H: %f S: %.2f, V: %.2f\n", H, S, V);
                    //printf("Iterations: %f\n", f_iterations);
                    //}
                    float h = H / 60;
                    float C = S * V;
                    float X = C * (1 - fabsf((fmodf(h, 2) - 1)));
                    float m = V - C;
                    float r, g, b;
                    if (h >= 0 && h <= 1) {
                        r = C;
                        g = X;
                        b = 0;
                    }
                    else if (h > 1 && h < 2) {
                        r = X;
                        g = C;
                        b = 0;
                    }
                    else if (h > 2 && h <= 3) {
                        r = 0;
                        g = C;
                        b = X;
                    }
                    else if (h > 3 && h <= 4) {
                        r = 0;
                        g = X;
                        b = C;
                    }
                    else if (h > 4 && h <= 5) {
                        r = X;
                        g = 0;
                        b = C;
                    }
                    else if (h > 5 && h <= 6) {
                        r = C;
                        g = 0;
                        b = X;
                    }
                    else { // color white to make stand out
                        r = 1 - m;
                        g = 1 - m;
                        b = 1 - m;
                    }
                    unsigned char red = (r + m) * 255;
                    unsigned char green = (g + m) * 255;
                    unsigned char blue = (b + m) * 255;
                    // End of conversion.

                    // Cap RGB values to 255
                    if (red > 255) {
                        red = 255;
                    }
                    if (green > 255) {
                        green = 255;
                    }
                    if (blue > 255) {
                        blue = 255;
                    }
                    pixel_color.r = red;
                    pixel_color.g = green;
                    pixel_color.b = blue;
                }
                (image->pixels_rgb)[3 * index + 0] = pixel_color.r; // Red value
                (image->pixels_rgb)[3 * index + 1] = pixel_color.g; // Green value
                (image->pixels_rgb)[3 * index + 2] = pixel_color.b; // Blue value
            }
        }
    }
}
