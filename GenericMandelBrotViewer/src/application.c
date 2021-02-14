#include <stdbool.h>
#include <glfw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

#include "mandelbrot_image.h"
#include "cuda_launch.h"
#include "constants.h"
#include "perftest.h"
#include "controls.h"
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

void allocate_memory(mandelbrot_image** image_ptr)
{
    size_t total_pixels = (size_t) (*image_ptr)->resolution_x * (*image_ptr)->resolution_y;
    if (g_cuda_device_available) {
        // Hack to get the memory address of the pointers.
        // We create a bunch of char** variables that hold the addresses of the pointers inside the mandelbrot_image_ptr struct, so that we can increment them by the sizes in bytes of the components of the struct. The reason their type is char** instead of just char* is that cudaMallocManaged takes void** as the first argument.
        char* byte_pointer = (char*)(*image_ptr);
        char** points_ptr = byte_pointer + 4 * sizeof(double) + 4 * sizeof(int); // 4 bytes "extra" because of struct padding, see https://stackoverflow.com/a/2749096/9069452
        char** iterated_points_ptr = (char*)points_ptr + sizeof(cuDoubleComplex*);
        char** squared_absolute_values_ptr = (char*)iterated_points_ptr + sizeof(cuDoubleComplex*);
        char** pixels_rgb_ptr = (char*)squared_absolute_values_ptr + sizeof(double*);
        char** iterationsArr_ptr = (char*)pixels_rgb_ptr + sizeof(char*);
        cudaMallocManaged(points_ptr, total_pixels * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
        cudaMallocManaged(iterated_points_ptr, total_pixels * sizeof(cuDoubleComplex), cudaMemAttachGlobal);
        cudaMallocManaged(squared_absolute_values_ptr, total_pixels * sizeof(double), cudaMemAttachGlobal);
        cudaMallocManaged(pixels_rgb_ptr, total_pixels * 3 * sizeof(unsigned char), cudaMemAttachGlobal);
        cudaMallocManaged(iterationsArr_ptr, total_pixels * sizeof(unsigned int), cudaMemAttachGlobal);
    } else if (!(g_cuda_device_available)) {
        (*image_ptr)->points = malloc(total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->iterated_points = malloc(total_pixels * sizeof(cuDoubleComplex));
        (*image_ptr)->squared_absolute_values = malloc(total_pixels * sizeof(double));
        (*image_ptr)->pixels_rgb = malloc(total_pixels * 3 * sizeof(unsigned char));
        (*image_ptr)->iterationsArr = malloc(total_pixels * sizeof(unsigned int));
    }
    // Check for NULL pointers:
    if (!(*image_ptr)->points || !(*image_ptr)->iterated_points || !(*image_ptr)->squared_absolute_values || !(*image_ptr)->pixels_rgb || !(*image_ptr)->iterationsArr) {
        printf("Not enough memory available, exiting.");
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
        printf("Not enough memory available, exiting.");
        exit(-1);
    }
}

void free_the_pointers(mandelbrot_image* image)
{
    if (g_cuda_device_available) {
        cudaFree(image->points);
        cudaFree(image->iterated_points);
        cudaFree(image->squared_absolute_values);
        cudaFree(image->pixels_rgb);
        cudaFree(image->iterationsArr);
    } else if (!(g_cuda_device_available)) {
        free(image->points);
        free(image->iterated_points);
        free(image->squared_absolute_values);
        free(image->pixels_rgb);
        free(image->iterationsArr);
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

void check_and_process_inputs(mandelbrot_image** image_ptr, mandelbrot_image* image, GLFWwindow* window)
{
    // Process keypresses if there were any:
    if (g_keypress_input_flag) {
        process_keyboard_input(g_last_keypress_input, image, window);
        reset_render_objects(image);
        g_keypress_input_flag = false;
    } else if (g_scroll_input_flag) {
        process_scroll_input(image, g_last_scroll_xoffset, g_last_scroll_yoffset);
        reset_render_objects(image);
        g_scroll_input_flag = false;
    } else if (g_resized_flag) {
        process_resize(image, window, g_resized_new_w, g_resized_new_h);
        reallocate_memory(image_ptr);
        reset_render_objects(image);
        g_resized_flag = false;
    }
}

void run_program_iteration(
    mandelbrot_image** image_ptr,
    mandelbrot_image* image,
    GLFWwindow* window,
    char* window_title,
    int g_iterations_per_frame)
{
    if (g_debugging_enabled) {
        //Sleep(500); // cap fps to 2
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
    check_and_process_inputs(image_ptr, image, window);
    if ((g_application_mode == MODE_VIEW || g_application_mode == MODE_PERFORMANCE_TEST) && g_rendered_iterations < image->max_iterations) {
        mandelbrot_iterate_and_color(image);
        g_rendered_iterations += g_iterations_per_frame;
        // Rename the window title
        sprintf(window_title, "GenericMandelbrotViewer | Center: %.4f + %.4f i | Max iterations: %d | Drawing radius: %.2f", image->center_real, image->center_imag, image->max_iterations, image->draw_radius);
        glfwSetWindowTitle(window, window_title);
        draw_pixels(image, window);
    }
}
