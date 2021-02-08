#pragma once
#include "starting_parameters.h"
#include "mandelbrot_image.h"

void process_scroll_input(mandelbrot_image* image, double xoffset, double yoffset) {
    if (yoffset > 0) {
        image->draw_radius *= 0.75; // zoom in
    }
    else if (yoffset < 0) {
        image->draw_radius /= 0.75; // zoom out
    }
}

void process_keyboard_input(int key, mandelbrot_image* image, GLFWwindow* window) {
    switch (key) {
    case GLFW_KEY_D:
        if (g_debugging_enabled) {
            g_debugging_enabled = false;
        }
        else {
            g_debugging_enabled = true;
        }
        break;
    case GLFW_KEY_EQUAL: // zoom in, = is also +
        image->draw_radius *= 0.75; // zoom in
        break;
    case GLFW_KEY_MINUS:
        image->draw_radius /= 0.75; // zoom out
        break;
    case GLFW_KEY_LEFT:
        image->center_real -= 0.1 * image->draw_radius;
        break;
    case GLFW_KEY_RIGHT:
        image->center_real += 0.1 * image->draw_radius;
        break;
    case GLFW_KEY_UP:
        image->center_imag += 0.1 * image->draw_radius;
        break;
    case GLFW_KEY_DOWN:
        image->center_imag -= 0.1 * image->draw_radius;
        break;
    case GLFW_KEY_LEFT_BRACKET:
        if (image->max_iterations > 2 && image->max_iterations < 10) {
            image->max_iterations--;
        }
        else if (image->max_iterations >= 10) {
            image->max_iterations *= 0.9;
        }
        printf("Max iterations now at: %d\n", image->max_iterations);
        if (incremental_iteration) {
            iterations_per_frame = incremental_iterations_per_frame;
        }
        else {
            iterations_per_frame = image->max_iterations;
        }
        break;
    case GLFW_KEY_RIGHT_BRACKET:
        if (image->max_iterations < 10) {
            image->max_iterations++;
        }
        else if (image->max_iterations >= 10) {
            image->max_iterations /= 0.9;
        }
        printf("Max iterations now at: %d\n", image->max_iterations);
        if (incremental_iteration) {
            iterations_per_frame = incremental_iterations_per_frame;
        }
        else {
            iterations_per_frame = image->max_iterations;
        }
        break;
        //case GLFW_KEY_I:
        //    if (incremental_iteration)
        //    {
        //        iterations_per_frame = incremental_iterations_per_frame;
        //        incremental_iteration = false;
        //    }
        //    else {
        //        iterations_per_frame = image->max_iterations;
        //        incremental_iteration = true;
        //    }
        //    break;
    case GLFW_KEY_ESCAPE:
        // Set the close flag of the window to TRUE so that the program exits:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_E:
        // Run a performance test:
        start_performance_test_flag = true;
        break;
    }
}

void process_resize(mandelbrot_image* image, GLFWwindow* window, int w, int h) {
    // Round numbers to multiples of 4:
    int maximum;
    int minimum;
    int new_width;
    if (w > h) {
        maximum = w;
        minimum = h;
    }
    else {
        maximum = h;
        minimum = w;
    }
    if (w <= image->resolution_x && h <= image->resolution_y) {
        // This happens when the user resizes by just dragging 
        // the bottom, top or sides; only w or h decreases so the
        // relevant size is the minimum.
        new_width = minimum- (minimum% 4);
    }
    else {
        new_width = maximum - (maximum % 4);
    }
    int new_height = new_width; // Image has to be square
    image->resolution_x = new_width;
    image->resolution_y = new_height;
    glfwSetWindowSize(window, new_width, new_height);
}