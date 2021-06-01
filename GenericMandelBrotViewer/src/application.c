#include <stdbool.h>
#include <glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include "util/color_palette.h"
#include "util/perftest.h"
#include "util/controls.h"
#include "mandelbrot_image.h"
#include "cuda_launch.h"
#include "constants.h"
#include "global.h"


void setup_incremental_iterations(mandelbrot_image* image)
{
    if (g_incremental_iteration) {
        g_iterations_per_frame = g_iterations_per_frame;
    }
    else {
        g_iterations_per_frame = image->max_iterations;
    }
}


void setup_colors() {
    set_color_rgb(&black, 0, 0, 0);
    set_color_rgb(&white, 255, 255, 255);
    set_color_rgb(&red, 255, 0, 0);
    set_color_rgb(&green, 0, 255, 0);
    set_color_rgb(&blue, 0, 0, 255);
    set_color_rgb(&blue_dark, 0, 0, 96);
    setup_palettes();
}


void free_the_pointers(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        if (image->points) cudaFree(image->points);
        if (image->iterated_points) cudaFree(image->iterated_points);
        if (image->squared_absolute_values) cudaFree(image->squared_absolute_values);
        if (image->pixels_rgb) cudaFree(image->pixels_rgb);
        if (image->iterationsArr) cudaFree(image->iterationsArr);
        if (image) cudaFree(image);
    }
    else if (!(g_cuda_device_available)) {
        if (image->points) free(image->points);
        if (image->iterated_points) free(image->iterated_points);
        if (image->squared_absolute_values) free(image->squared_absolute_values);
        if (image->pixels_rgb) free(image->pixels_rgb);
        if (image->iterationsArr) free(image->iterationsArr);
        if (image) free(image);
    }
}


void allocate_memory(mandelbrot_image** image_ptr)
{
    size_t total_pixels = (size_t)(*image_ptr)->resolution_x * (*image_ptr)->resolution_y;
    if (g_cuda_device_available) {
        cudaMallocManaged(&((*image_ptr)->points), total_pixels * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
        cudaMallocManaged(&((*image_ptr)->iterated_points), total_pixels * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
        cudaMallocManaged(&((*image_ptr)->squared_absolute_values), total_pixels * sizeof(double), cudaMemAttachGlobal);
        cudaMallocManaged(&((*image_ptr)->pixels_rgb), total_pixels * 3 * sizeof(unsigned char), cudaMemAttachGlobal);
        cudaMallocManaged(&((*image_ptr)->iterationsArr), total_pixels * sizeof(unsigned int), cudaMemAttachGlobal);
    }
    else if (!(g_cuda_device_available)) {
        (*image_ptr)->points = malloc(total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->iterated_points = malloc(total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->squared_absolute_values = malloc(total_pixels * sizeof(double));
        (*image_ptr)->pixels_rgb = malloc(total_pixels * 3 * sizeof(unsigned char));
        (*image_ptr)->iterationsArr = malloc(total_pixels * sizeof(unsigned int));
    }
    // Check for NULL pointers:
    if (!(*image_ptr)->points || !(*image_ptr)->iterated_points || !(*image_ptr)->squared_absolute_values || !(*image_ptr)->pixels_rgb || !(*image_ptr)->iterationsArr) {
        printf("Not enough memory available on allocation, exiting.");
        free_the_pointers(*image_ptr);
        exit(-1);
    }
}


void reallocate_memory(mandelbrot_image** image_ptr)
{
    size_t total_pixels = (size_t) (*image_ptr)->resolution_x * (*image_ptr)->resolution_y;
    if (g_cuda_device_available) {
        // Reallocation isn't a thing in CUDA, so we'll just free the
        // memory and then allocate memory again.
        cudaFree((*image_ptr)->points);
        cudaFree((*image_ptr)->iterated_points);
        cudaFree((*image_ptr)->squared_absolute_values);
        cudaFree((*image_ptr)->pixels_rgb);
        cudaFree((*image_ptr)->iterationsArr);
        allocate_memory(image_ptr);
    } else if (!(g_cuda_device_available)) {
        (*image_ptr)->points = realloc((*image_ptr)->points, total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->iterated_points = realloc((*image_ptr)->iterated_points, total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->squared_absolute_values = realloc((*image_ptr)->squared_absolute_values, total_pixels * sizeof(double));
        (*image_ptr)->pixels_rgb = realloc((*image_ptr)->pixels_rgb, total_pixels * 3 * sizeof(unsigned char));
        (*image_ptr)->iterationsArr = realloc((*image_ptr)->iterationsArr, total_pixels * sizeof(unsigned int));
    }
    // Check for NULL pointers:
    if (!(*image_ptr)->points || !(*image_ptr)->iterated_points || !(*image_ptr)->squared_absolute_values || !(*image_ptr)->pixels_rgb || !(*image_ptr)->iterationsArr) {
        printf("Not enough memory available on reallocation, exiting.");
        free_the_pointers(*image_ptr);
        exit(-1);
    }
}


void draw_pixels(mandelbrot_image* image, GLFWwindow* window)
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(image->resolution_x, image->resolution_y, GL_RGB, GL_UNSIGNED_BYTE, image->pixels_rgb);
    // Swap front and back buffers
    glfwSwapBuffers(window);
}

int setup_glfw(GLFWwindow* window)
{
    if (!window) {
        glfwTerminate();
        return -1;
    }
    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, window_callback);
    return 1;
}

void check_and_process_inputs(mandelbrot_image** image_ptr, mandelbrot_image* image, GLFWwindow* window, GLFWmonitor* monitor)
{
    // Process keypresses if there were any:
    if (g_keypress_input_flag) {
        process_keyboard_input(g_last_keypress_input, image, window, monitor);
        reset_render_objects(image);
        g_keypress_input_flag = false;
    } else if (g_scroll_input_flag) {
        process_scroll_input(image, g_last_scroll_xoffset, g_last_scroll_yoffset);
        reset_render_objects(image);
        g_scroll_input_flag = false;
    } else if (g_resized_flag) {
        process_resize(image_ptr, image, window, monitor, g_resized_new_w, g_resized_new_h);
        g_resized_flag = false;
    }
}

void create_highres_image(mandelbrot_image** image_ptr, mandelbrot_image* image, int res_x, int res_y) {
    //printf("Creating high resolution image...");
    int iterations_temp = g_rendered_iterations;
    int resolution_x_temp = image->resolution_x;
    int resolution_y_temp = image->resolution_y;
    image->resolution_x = res_x;
    image->resolution_y = res_y;
    reallocate_memory(image_ptr);
    reset_render_objects(image);
    mandelbrot_iterate_n_and_color(image, image->max_iterations);
    printf("Creating image `mandelbrot.png`.\n");
    create_png("mandelbrot.png", image->resolution_x, image->resolution_y, image->pixels_rgb);
    printf("Done creating image.\n");
    image->resolution_x = resolution_x_temp;
    image->resolution_y = resolution_y_temp;
    reallocate_memory(image_ptr);
    reset_render_objects(image);
}

void run_program_iteration(
    mandelbrot_image** image_ptr,
    mandelbrot_image* image,
    GLFWwindow* window,
    GLFWmonitor* monitor,
    char* window_title,
    int g_iterations_per_frame)
{
    if (g_debugging_enabled) {
        //Sleep(500); // cap fps to 2
    }
    if (g_create_image_flag) {
        create_highres_image(image_ptr, image, 4096, 4096);
        g_create_image_flag = false;
    }
    if (g_start_performance_test_flag) {
        g_application_mode = MODE_PERFORMANCE_TEST;
        start_performance_test(image);
    }
    if (g_application_mode == MODE_PERFORMANCE_TEST) {
        setup_performance_iteration(image);
        g_performance_iterations_done++;
        if (g_performance_iterations_done >= g_performance_iterations_total) {
            g_application_mode = MODE_VIEW;
            printf("\rPerformance test took %d ms.           \n", end_performance_test(image));
            fflush(stdout);
        }
    }
    check_and_process_inputs(image_ptr, image, window, monitor);
    if ((g_application_mode == MODE_VIEW || g_application_mode == MODE_PERFORMANCE_TEST) && g_rendered_iterations < image->max_iterations) {
        mandelbrot_iterate_n_and_color(image, image->max_iterations);
        g_rendered_iterations += image->max_iterations;
        // Rename the window title
        sprintf(window_title, "GenericMandelbrotViewer | Center: %.4f + %.4f i | Max iterations: %d | Drawing radius (horizontal): %.2f", image->center_real, image->center_imag, image->max_iterations, image->draw_radius_x);
        glfwSetWindowTitle(window, window_title);
        draw_pixels(image, window);
    }
}
