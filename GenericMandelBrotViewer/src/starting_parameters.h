#pragma once
#include <stdbool.h>
#include <math.h>

#include "mandelbrot_image.h"

// Debugging related parameters:
bool g_debugging_enabled;
int performance_iterations_total;
int performance_iterations_done;
bool start_performance_test_flag;

void setup_debugging_performance_parameters() {
	g_debugging_enabled = false;
	start_performance_test_flag = false;
	performance_iterations_done = 0;
	performance_iterations_total = 256;
}

void setup_image_parameters(mandelbrot_image* image) {
	image->center_real = 0.0;
	image->center_imag = 0.0;
	// For some (currently unknown) reason, these have to be multiples of four.
	image->resolution_x = 516;
	image->resolution_y = 516;
	image->draw_radius = 2.5;
	image->escape_radius_squared = 4;
	image->max_iterations = 64;
}

// Behavioral parameters:
#define MODE_VIEW 0
#define MODE_PERFORMANCE_TEST 1
unsigned short g_application_mode;
bool incremental_iteration;
int iterations_per_frame;
int incremental_iterations_per_frame;
int rendered_iterations;

void setup_behavioral_parameters() {
	g_application_mode = MODE_VIEW;
	incremental_iteration = false;
	iterations_per_frame = 1; // also set later based on whether incremental iterations are enabled.
	incremental_iterations_per_frame = 4;
	rendered_iterations = 0;
}

// Cuda parameters:
int cuda_block_size;
int cuda_num_blocks;
bool cuda_device_available;

void setup_cuda_paramaters(mandelbrot_image* image) {
	cuda_block_size = 256;
	cuda_num_blocks = (int) ceil(image->resolution_x * image->resolution_y / cuda_block_size);
	cuda_device_available = false;
}

void setup_starting_parameters(mandelbrot_image* image) {
	setup_image_parameters(image);
	setup_behavioral_parameters();
	setup_debugging_performance_parameters();
	setup_cuda_paramaters(image);
}
