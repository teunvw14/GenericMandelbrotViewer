#pragma once

#include <glfw3.h>

#include "mandelbrot_image.h"
#include "global.h"

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void window_callback(GLFWwindow* window, int w, int h);
void process_scroll_input(mandelbrot_image* image, double xoffset, double yoffset);
void process_keyboard_input(int key, mandelbrot_image* image, GLFWwindow* window);
void process_resize(mandelbrot_image* image, GLFWwindow* window, int w, int h);
