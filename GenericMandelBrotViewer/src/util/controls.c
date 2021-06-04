#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <glfw3.h>

#include "../mandelbrot_image.h"
#include "../cuda_launch.h"
#include "../application.h"
#include "../constants.h"
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

void mouse_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_1) {
        g_lmb_input_flag = true;
        g_lmb_input_action = action;
    }
}

void cursor_move_callback(GLFWwindow* window, double xpos, double ypos) {
    g_cursor_moved = true;
    g_cursor_pos_x = xpos;
    g_cursor_pos_y = ypos;
}


void process_scroll_input(mandelbrot_image* image, double xoffset, double yoffset)
{
    // Don't do anything with scrolling / zooming when doing a performance test
    if (g_application_mode == MODE_PERFORMANCE_TEST) {
        return;
    }
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
        //printf("Max iterations now at: %d\n", image->max_iterations);
        break;
    case GLFW_KEY_RIGHT_BRACKET:
        if (image->max_iterations < 10) {
            image->max_iterations++;
        }
        else if (image->max_iterations >= 10) {
            image->max_iterations /= 0.9;
        }
        //printf("Max iterations now at: %d\n", image->max_iterations);
        break;
    case GLFW_KEY_ESCAPE:
        // Set the close flag of the window to TRUE so that the program exits:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_E:
        // Start performance test (if one isn't already running):
        if (g_application_mode != MODE_PERFORMANCE_TEST) {
            g_start_performance_test_flag = true;
        }
        break;
    case GLFW_KEY_P:
        g_create_image_flag = true;
        break;
    case GLFW_KEY_M:
        g_coloring_mode += 1;
        g_coloring_mode %= 3;
        break;
    case GLFW_KEY_SPACE: {
        for (int i = 0; i < palette_pretty.length; i++) {
            palette_pretty.colors[i] = palette_pretty.colors[(i + 1) % palette_pretty.length];
        }
        break;
    }
    case GLFW_KEY_U:
        if (g_lowres_block_size < 64) { g_lowres_block_size++; }
        break;
    case GLFW_KEY_I:
        if (g_lowres_block_size > 2) { g_lowres_block_size--; }
        break;
    case GLFW_KEY_ENTER: {
        printf("Current coordinates: (%.16f, %.16f)\n", image->center_real, image->center_imag);
        char new_x_str[32];
        char new_y_str[32];
        float new_x, new_y;
        printf("New x coordinate: ");
        fgets(new_x_str, 32, stdin);
        printf("New y coordinate: ");
        fgets(new_y_str, 32, stdin);
        new_x = strtod(new_x_str, NULL);
        new_y = strtod(new_y_str, NULL);
        image->center_real = new_x;
        image->center_imag = new_y;
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
    if (w >= monitor_width || h >= monitor_height) {
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
    //printf("Resizing to w: %d h: %d\n", new_width, new_height);
    image->resolution_x = new_width;
    image->resolution_y = new_height;
    glfwSetWindowSize(window, new_width, new_height);
    reallocate_memory(image_ptr);
    reset_render_objects(image);
}

void process_mouse_input(mandelbrot_image* image, GLFWwindow* window, int action) {
    if (action == GLFW_PRESS) {
        g_lmb_pressed = true;
        glfwGetCursorPos(window, &g_dragging_start_x, &g_dragging_start_y);
        g_dragging_center_real = image->center_real;
        g_dragging_center_imag = image->center_imag;
    } else if (action == GLFW_RELEASE) {
        g_lmb_pressed = false;
    }
}

void process_cursor_move(mandelbrot_image* image, GLFWwindow* window, double xpos, double ypos) {
    if (g_lmb_pressed && g_application_mode == MODE_VIEW) {
        image->center_real = g_dragging_center_real + 2 * image->draw_radius_x * (g_dragging_start_x - xpos)/ image->resolution_x;
        image->center_imag = g_dragging_center_imag + 2 * image->draw_radius_y * (ypos - g_dragging_start_y) / image->resolution_y;
        reset_render_objects(image);
    }
}
