#pragma once
#include <stdbool.h>
#include <sys/timeb.h>

#include "util/color_palette.h"

// Cuda
int g_cuda_block_size;
int g_cuda_num_blocks;
bool g_cuda_device_available;

// Debugging / performance testing
bool g_debugging_enabled;
int g_performance_iterations_total;
int g_performance_iterations_done;
bool g_start_performance_test_flag;
struct timeb g_start, g_end;
int g_max_iterations_store;
int g_center_real_store;
int g_center_imag_store;
double g_draw_radius_x_store;
double g_draw_radius_y_store;

// Behavioral parameters
unsigned short g_application_mode;
unsigned short g_coloring_mode;
palette g_coloring_palette;
bool g_incremental_iteration;
bool g_create_image_flag;
int g_iterations_per_frame;
int g_incremental_iterations_per_frame;
int g_rendered_iterations;

// Input related
bool g_keypress_input_flag;
int g_last_keypress_input;
bool g_scroll_input_flag;
int g_last_scroll_xoffset;
int g_last_scroll_yoffset;
bool g_resized_flag;
int g_resized_new_w;
int g_resized_new_h;
