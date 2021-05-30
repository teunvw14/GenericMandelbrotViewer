#pragma once
#include <glfw3.h>
#include "mandelbrot_image.h"


int setup_glfw(GLFWwindow* window);
void setup_colors();
void allocate_memory(mandelbrot_image** image);
void reallocate_memory(mandelbrot_image** image);
void free_the_pointers(mandelbrot_image* image);
void setup_incremental_iterations(mandelbrot_image* image);
void draw_pixels(mandelbrot_image* image, GLFWwindow* window);
void run_program_iteration(mandelbrot_image** image_ptr, mandelbrot_image* image, GLFWwindow* window, GLFWmonitor* monitor, char* window_title, int g_iterations_per_frame);
void check_and_process_inputs(mandelbrot_image** image_ptr, mandelbrot_image* image, GLFWwindow* window, GLFWmonitor* monitor);
void create_highres_image(mandelbrot_image** image_ptr, mandelbrot_image* image, int res_x, int res_y);
