#pragma once

#include <stdio.h>

#include <glfw3.h>

#include "../mandelbrot_image.h"
#include "../cuda_launch.h"
#include "../application.h"
#include "../global.h"
#include "color_palette.h"
#include "create_png.h"


// The callbacks are only for setting flags. The inputs are actually processed 
// later, inside the `process_x` functions at the bottom of this file.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS) {
        g_keypress_input_flag = true;
        g_last_keypress_input = key;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_scroll_input_flag = true;
    g_last_scroll_xoffset = xoffset;
    g_last_scroll_yoffset = yoffset;
}

void window_callback(GLFWwindow* window, int w, int h)
{
    g_resized_flag = true;
    g_resized_new_w = w;
    g_resized_new_h = h;
}

void process_scroll_input(mandelbrot_image* image, double xoffset, double yoffset)
{
    if (yoffset > 0) {
        image->draw_radius_x *= 0.75; // zoom in
        image->draw_radius_y *= 0.75;
    }
    else if (yoffset < 0) {
        image->draw_radius_x /= 0.75; // zoom out
        image->draw_radius_y /= 0.75;
    }
}

void process_keyboard_input(int key, mandelbrot_image* image, GLFWwindow* window, GLFWmonitor* monitor)
{
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
        image->draw_radius_x *= 0.75; // zoom in
        image->draw_radius_y *= 0.75;
        break;
    case GLFW_KEY_MINUS:
        image->draw_radius_x /= 0.75; // zoom out
        image->draw_radius_y /= 0.75;
        break;
    case GLFW_KEY_LEFT:
        image->center_real -= 0.1 * image->draw_radius_x;
        break;
    case GLFW_KEY_RIGHT:
        image->center_real += 0.1 * image->draw_radius_x;
        break;
    case GLFW_KEY_UP:
        image->center_imag += 0.1 * image->draw_radius_y;
        break;
    case GLFW_KEY_DOWN:
        image->center_imag -= 0.1 * image->draw_radius_y;
        break;
    case GLFW_KEY_LEFT_BRACKET:
        if (image->max_iterations > 2 && image->max_iterations < 10) {
            image->max_iterations--;
        }
        else if (image->max_iterations >= 10) {
            image->max_iterations *= 0.9;
        }
        printf("Max iterations now at: %d\n", image->max_iterations);
        if (g_incremental_iteration) {
            g_iterations_per_frame = g_incremental_iterations_per_frame;
        }
        else {
            g_iterations_per_frame = image->max_iterations;
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
        if (g_incremental_iteration) {
            g_iterations_per_frame = g_incremental_iterations_per_frame;
        }
        else {
            g_iterations_per_frame = image->max_iterations;
        }
        break;
        //case GLFW_KEY_I:
        //    if (g_incremental_iteration)
        //    {
        //        g_iterations_per_frame = g_incremental_iterations_per_frame;
        //        g_incremental_iteration = false;
        //    }
        //    else {
        //        g_iterations_per_frame = image->max_iterations;
        //        g_incremental_iteration = true;
        //    }
        //    break;
    case GLFW_KEY_ESCAPE:
        // Set the close flag of the window to TRUE so that the program exits:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_E:
        // Run a performance test:
        g_start_performance_test_flag = true;
        break;
    case GLFW_KEY_P:
        g_create_image_flag = true;
        break;
    case GLFW_KEY_M:
        g_coloring_mode += 1;
        g_coloring_mode %= 3;
        break;
    case GLFW_KEY_SPACE: {
        color_rgb first = palette_pretty.colors[0];
        for (int i = 0; i < palette_pretty.length; i++) {
            palette_pretty.colors[i] = palette_pretty.colors[(i + 1) % palette_pretty.length];
        }
        break;
    }
    }
}

void process_resize(mandelbrot_image** image_ptr, mandelbrot_image* image, GLFWwindow* window, GLFWmonitor* monitor, int w, int h)
{
    int new_width;
    int new_height;
    int maximum;
    int minimum;
    int monitor_width, monitor_height;
    glfwGetMonitorWorkarea(monitor, NULL, NULL, &monitor_width, &monitor_height);
    printf("Available space - w: %d h: %d\n", monitor_width, monitor_height);
    if (w >= monitor_width || h >= monitor_height) {
        printf("Was here\n");
        new_width = monitor_width - (monitor_width % 4);
        new_height = monitor_height - (monitor_height % 4);
        image->draw_radius_x *= (double) new_width / (double) new_height;
    }
    else {
        if (w > h) {
            maximum = w;
            minimum = h;
        }
        else {
            maximum = h;
            minimum = w;
        }
        // Round numbers to multiples of 4:
        if (w <= image->resolution_x && h <= image->resolution_y) {
            // This happens when the user resizes by just dragging
            // the bottom, top or sides; only w or h decreases so the
            // relevant size is the minimum.
            new_width = minimum - (minimum % 4);
        }
        else {
            new_width = maximum - (maximum % 4);
        }
        // Image has to be square
        new_height = new_width; 
        image->draw_radius_x = image->draw_radius_y;
    }
    printf("Resizing to w: %d h: %d\n", new_width, new_height);
    image->resolution_x = new_width;
    image->resolution_y = new_height;
    glfwSetWindowSize(window, new_width, new_height);
    reallocate_memory(image_ptr);
    reset_render_objects(image);
}