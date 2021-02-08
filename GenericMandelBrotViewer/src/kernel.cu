#include <glfw3.h>
#include <stdio.h>
#include <math.h>

// CUDA imports
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

// Debugging and profiling:
#include <sys/timeb.h>
#include <windows.h>
struct timeb start, end;
int max_iterations_store;
int center_real_store;
int center_imag_store;
double draw_radius_store;

#include "controls.h"
#include "mandelbrot_image.h"
#include "starting_parameters.h"

bool g_keypress_input_flag;
int g_last_keypress_input;
bool g_scroll_input_flag;
int g_last_scroll_xoffset;
int g_last_scroll_yoffset;
bool g_resized_flag;
int g_resized_new_w;
int g_resized_new_h;

__global__ void build_complex_grid_cuda(mandelbrot_image* image) {
    // Create a grid of complex numbers around the center point (center_real, center_imag).
    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;
    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;

    double step_x = 2 * image->draw_radius / image->resolution_x;
	double step_y = 2 * image->draw_radius / image->resolution_y;
	double point_re;
	double point_im;
	int index;

	// Start drawing in the bottom left, go row by row.
	for (int pixel_y = block_index_x; pixel_y < image->resolution_y; pixel_y += block_stride_x)
	{
		point_im = image->center_imag + pixel_y * step_y - image->draw_radius;
		for (int pixel_x = thread_index_x; pixel_x < image->resolution_x; pixel_x += thread_stride_x)
		{
			index = pixel_y * image->resolution_y + pixel_x;
			point_re = image->center_real + pixel_x * step_x - image->draw_radius;
            image->points[index] = make_cuDoubleComplex(point_re, point_im);
            image->iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
		}
	}
}

void build_complex_grid_non_cuda(mandelbrot_image* image) {
    // Create a grid of complex numbers around the center point (center_real, center_imag).

    double step_x = 2 * image->draw_radius / image->resolution_x;
    double step_y = 2 * image->draw_radius / image->resolution_y;
    double point_re;
    double point_im;
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++)
    {
        point_im = image->center_imag + pixel_y * step_y - image->draw_radius;
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++)
        {
            index = pixel_y * image->resolution_y + pixel_x;
            point_re = image->center_real + pixel_x * step_x - image->draw_radius;
            image->points[index] = make_cuDoubleComplex(point_re, point_im);
            image->iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
        }
    }
}

__global__ void reset_render_arrays_cuda(mandelbrot_image* image) {
    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;
    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;
    int res_x = image->resolution_x;
    int res_y = image->resolution_y;
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = block_index_x; pixel_y < res_y; pixel_y += block_stride_x)
    {
        for (int pixel_x = thread_index_x; pixel_x < res_x; pixel_x += thread_stride_x)
        {
            index = pixel_y * image->resolution_y + pixel_x;
            image->iterationsArr[index] = 0;
            image->squared_absolute_values[index] = 0;
        }
    }
}

void reset_render_arrays_non_cuda(mandelbrot_image* image) {
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++)
    {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++)
        {
            index = pixel_y * image->resolution_y + pixel_x;
            image->iterationsArr[index] = 0;
            image->squared_absolute_values[index] = 0;
        }
    }
}

__global__ void mandelbrot_iterate_cuda(mandelbrot_image* image) {
    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;

    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;
    int index;

    for (int pixel_y = block_index_x; pixel_y < image->resolution_y; pixel_y += block_stride_x)
    {
        for (int pixel_x = thread_index_x; pixel_x < image->resolution_x; pixel_x += thread_stride_x)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_y + pixel_x;
            cuDoubleComplex starting_number = image->points[index];
            cuDoubleComplex iterated_point = image->iterated_points[index];
            double sq_abs = image->squared_absolute_values[index];
            unsigned int iterations_ = image->iterationsArr[index];
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

void mandelbrot_iterate_non_cuda(mandelbrot_image* image) {
    int index = 0;

    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++)
    {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_y + pixel_x;
            cuDoubleComplex starting_number = image->points[index];
            cuDoubleComplex iterated_point = image->iterated_points[index];
            double sq_abs = image->squared_absolute_values[index];
            unsigned int iterations_ = image->iterationsArr[index];
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


__global__ void color_cuda(mandelbrot_image* image) {
    // Do some coloring!

    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;

    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;
    int index;
    unsigned int iterations;

    for (int pixel_y = block_index_x; pixel_y < image->resolution_y; pixel_y += block_stride_x)
    {
        for (int pixel_x = thread_index_x; pixel_x < image->resolution_x; pixel_x += thread_stride_x)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_y + pixel_x;
            iterations = image->iterationsArr[index];
            if (iterations == image->max_iterations)
            {
                // Values that don't escape are colored black:
                image->pixels_rgb[3 * index + 0] = 0; // Red value
                image->pixels_rgb[3 * index + 1] = 0; // Green value
                image->pixels_rgb[3 * index + 2] = 0; // Blue value
            } 
            else
            {
                float f_iterations = (float)iterations;
                float f_max_iterations = (float)image->max_iterations;
                // Smooth colors!
                float escape_size = __double2float_rn(image->squared_absolute_values[index]);
                float smoothed_iterations = iterations + 1 - log2f(log(escape_size)) + sqrtf(sqrtf(image->draw_radius));
                float H = 360*smoothed_iterations / f_max_iterations;
                float S = .65;
                float V = 1;


                // HSV to RGB conversion, yay!
                // TODO: look into edge cases for H and why they happen.
                //if (H > 360 || H < 0 || S > 1 || S < 0 || V > 1 || V < 0)
                //{
                    //printf("x");
                    //printf("The given HSV values are not in valid range.\n H: %f S: %.2f, V: %.2f\n", H, S, V);
                    //printf("Iterations: %f\n", f_iterations);
                //}
                float h = H / 60;
                float C = S * V;
                float X = C * (1 - fabsf((fmodf(h, 2) - 1)));
                float m = V - C;
                float r, g, b;
                if (h >= 0 && h <= 1)
                {
                    r = C;
                    g = X;
                    b = 0;
                }
                else if (h > 1 && h < 2)
                {
                    r = X;
                    g = C;
                    b = 0;
                }
                else if (h > 2 && h <= 3)
                {
                    r = 0;
                    g = C;
                    b = X;
                }
                else if (h > 3 && h <= 4)
                {
                    r = 0;
                    g = X;
                    b = C;
                }
                else if (h > 4 && h <= 5)
                {
                    r = X;
                    g = 0;
                    b = C;
                }
                else if (h > 5 && h <= 6)
                {
                    r = C;
                    g = 0;
                    b = X;
                }
                else // color white to make stand out
                {
                    r = 1-m;
                    g = 1-m;
                    b = 1-m;
                }
                unsigned char red = (r + m) * 255;
                unsigned char green = (g + m) * 255;
                unsigned char blue = (b + m) * 255;
                // End of conversion.

                // Cap RGB values to 255
                if (red > 255) { red = 255; }
                if (green > 255) { green = 255; }
                if (blue > 255) { blue = 255; }

                image->pixels_rgb[3 * index + 0] = red; // Red value
                image->pixels_rgb[3 * index + 1] = green; // Green value
                image->pixels_rgb[3 * index + 2] = blue; // Blue value
            }
        }
    }
}


void color_non_cuda(mandelbrot_image* image){

    // Do some coloring!
    int index;
    unsigned int iterations;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = 0; pixel_y < image->resolution_y; pixel_y++)
    {
        for (int pixel_x = 0; pixel_x < image->resolution_x; pixel_x++)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * image->resolution_y + pixel_x;
            iterations = image->iterationsArr[index];
            if (iterations == image->max_iterations)
            {
                // Values that don't escape are colored black:
                image->pixels_rgb[3 * index + 0] = 0; // Red value
                image->pixels_rgb[3 * index + 1] = 0; // Green value
                image->pixels_rgb[3 * index + 2] = 0; // Blue value
            }
            else
            {
                float f_iterations = (float)iterations;
                float f_max_iterations = (float)image->max_iterations;
                // Smooth colors!
                float escape_size = (float)(image->squared_absolute_values[index]);
                float smoothed_iterations = iterations + 1 - log2f(log(escape_size)) + sqrtf(sqrtf(image->draw_radius));
                float H = 360 * smoothed_iterations / f_max_iterations;
                float S = .65;
                float V = 1;


                // HSV to RGB conversion, yay!
                // TODO: look into edge cases for H and why they happen.
                //if (H > 360 || H < 0 || S > 1 || S < 0 || V > 1 || V < 0)
                //{
                    //printf("x");
                    //printf("The given HSV values are not in valid range.\n H: %f S: %.2f, V: %.2f\n", H, S, V);
                    //printf("Iterations: %f\n", f_iterations);
                //}
                float h = H / 60;
                float C = S * V;
                float X = C * (1 - fabsf((fmodf(h, 2) - 1)));
                float m = V - C;
                float r, g, b;
                if (h >= 0 && h <= 1)
                {
                    r = C;
                    g = X;
                    b = 0;
                }
                else if (h > 1 && h < 2)
                {
                    r = X;
                    g = C;
                    b = 0;
                }
                else if (h > 2 && h <= 3)
                {
                    r = 0;
                    g = C;
                    b = X;
                }
                else if (h > 3 && h <= 4)
                {
                    r = 0;
                    g = X;
                    b = C;
                }
                else if (h > 4 && h <= 5)
                {
                    r = X;
                    g = 0;
                    b = C;
                }
                else if (h > 5 && h <= 6)
                {
                    r = C;
                    g = 0;
                    b = X;
                }
                else // color white to make stand out
                {
                    r = 1 - m;
                    g = 1 - m;
                    b = 1 - m;
                }
                unsigned char red = (r + m) * 255;
                unsigned char green = (g + m) * 255;
                unsigned char blue = (b + m) * 255;
                // End of conversion.

                // Cap RGB values to 255
                if (red > 255) { red = 255; }
                if (green > 255) { green = 255; }
                if (blue > 255) { blue = 255; }

                image->pixels_rgb[3 * index + 0] = red; // Red value
                image->pixels_rgb[3 * index + 1] = green; // Green value
                image->pixels_rgb[3 * index + 2] = blue; // Blue value
            }
        }
    }
}

void build_complex_grid(mandelbrot_image* image)
{
    if (cuda_device_available) {
        build_complex_grid_cuda <<< cuda_num_blocks, cuda_block_size >>> (image);
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)){
        build_complex_grid_non_cuda(image);
    }
}

void mandelbrot_iterate_and_color(mandelbrot_image* image)
{
    if (cuda_device_available) {
        mandelbrot_iterate_cuda <<< cuda_num_blocks, cuda_block_size >>> (image);
        cudaDeviceSynchronize(); // TODO: maybe remove this? perhaps possible speedup
        color_cuda <<< cuda_num_blocks, cuda_block_size >>> (image);
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)) {
        mandelbrot_iterate_non_cuda(image);
        color_non_cuda(image);
    }
}

// Under maintenance.
void mandelbrot_iterate_n_and_color(mandelbrot_image* image, int iterations)
{
    mandelbrot_iterate_and_color(image);
    //mandelbrot_iterate_and_color_cuda << < cuda_num_blocks, cuda_block_size >> > (iterations, escape_radius_squared, resolution_x, image->resolution_y, points, iterated_points, squared_absolute_values, pixels_rgb);
}


void reset_render_objects(mandelbrot_image* image)
{
    // This function resets all the variables that are used for rendering the Mandelbrot. 

    size_t total_pixels = image->resolution_x * image->resolution_y;
    // Reset the `squared_absolute_values` to zero by allocating the memory space again.
    if (cuda_device_available) {
        reset_render_arrays_cuda <<< 1, 1024 >>> (image);
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)) {
        reset_render_arrays_non_cuda(image);
    }
    // Rebuild the grid of complex numbers based on (new) center_real and (new) center_imag.
    build_complex_grid(image);

    // Reset the amount of rendered iterations to 0. 
    rendered_iterations = 0;
}

// Performance testing functions below:
void start_performance_test(mandelbrot_image* image) {
    printf("Starting performance test.\n");
    start_performance_test_flag = false;
    performance_iterations_done = 0;
    rendered_iterations = 0;
    reset_render_objects(image);
    ftime(&start);

    // Set parameters for test:
    max_iterations_store = image->max_iterations;
    center_real_store = image->center_real;
    center_imag_store = image->center_imag;
    draw_radius_store = image->draw_radius;
}

void setup_performance_iteration(mandelbrot_image* image) {

    int starting_max_iterations;
    // Choose a spot to move to, and change starting max_iterations accordingly
    unsigned short spot = ceil(4.0 * (float) performance_iterations_done / (float) performance_iterations_total);
    if (spot <= 0) { spot = 1; }
    switch (spot) {
    case 1:
        image->center_real = -1.769249938555972345710642912303869;
        image->center_imag = -0.05694208981877081632294590463061;
        image->draw_radius = 1.9 * pow(10, -13);
        starting_max_iterations = 512;
        break;
    case 2:
        image->center_real = -0.0452407411;
        image->center_imag = 0.9868162204352258;
        image->draw_radius = 4.4 * pow(10, -9);
        starting_max_iterations = 128;
        break;
    case 3:
        image->center_real = -0.7336438924199521;
        image->center_imag = 0.2455211406714035;
        image->draw_radius = 4.5 * pow(10, -14);
        starting_max_iterations = 624;
        break;
    case 4:
        image->center_real = -0.0452407411;
        image->center_imag = 0.9868162204352258;
        image->draw_radius = 4.4 * pow(10, -9);
        starting_max_iterations = 128;
        break;
    default:
        break;
    }
    // hack to start iterating from `starting_max_iterations` for each new spot
    image->max_iterations = starting_max_iterations + performance_iterations_done * 4 - ((performance_iterations_total - 1) * 4 * (spot - 1)/ 4);
    printf("\rRendering spot %d with %d iterations.", spot, image->max_iterations); fflush(stdout);

    reset_render_objects(image);
}

int end_performance_test(mandelbrot_image* image) {
    ftime(&end);
    image->max_iterations = max_iterations_store;
    image->center_real = center_real_store;
    image->center_imag = center_imag_store;
    image->draw_radius = draw_radius_store;
    reset_render_objects(image);
    int elapsed_time = (int)1000.0 * (end.time - start.time) + (end.millitm - start.millitm);
    return elapsed_time;
}

// TODO: Move parsing of key inputs to separate function (that happens before rendering) so that the `G_IMAGE` variable can be thrown out.
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        g_keypress_input_flag = true;
        g_last_keypress_input = key;
    }
}

// TODO: Move parsing of scroll inputs to separate function (that happens before rendering) so that the `G_IMAGE` variable can be thrown out.
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    g_scroll_input_flag = true;
    g_last_scroll_xoffset = xoffset;
    g_last_scroll_yoffset = yoffset;
}

void window_callback(GLFWwindow* window, int w, int h) {
    g_resized_flag = true;
    g_resized_new_w = w;
    g_resized_new_h = h;
    // Resize memory blocks

}

void allocate_memory(mandelbrot_image* image) {
    size_t total_pixels = image->resolution_x * image->resolution_y;
    if (cuda_device_available) {
        cudaMallocManaged(&(image->points), total_pixels * sizeof(cuDoubleComplex));
        cudaMallocManaged(&(image->iterated_points), total_pixels * sizeof(cuDoubleComplex));
        cudaMallocManaged(&(image->squared_absolute_values), total_pixels * sizeof(double));
        cudaMallocManaged(&(image->pixels_rgb), total_pixels * 3 * sizeof(unsigned char));
        cudaMallocManaged(&(image->iterationsArr), total_pixels * sizeof(unsigned int));
    }
    else if (!(cuda_device_available)) {
        image->points = (cuDoubleComplex *)malloc(total_pixels * sizeof(cuDoubleComplex));
        image->iterated_points = (cuDoubleComplex *)malloc(total_pixels * sizeof(cuDoubleComplex));
        image->squared_absolute_values = (double*)malloc(total_pixels * sizeof(double));
        image->pixels_rgb = (unsigned char*)malloc(total_pixels * 3 * sizeof(unsigned char));
        image->iterationsArr = (unsigned int*)malloc(total_pixels * sizeof(unsigned int));
    }
}

void reallocate_memory(mandelbrot_image* image) {
    size_t total_pixels = image->resolution_x * image->resolution_y;
    if (cuda_device_available) {
        // Reallocation isn't a thing in CUDA, so we'll just free the
        // memory and then allocate memory again.
        cudaFree(image->points);
        cudaFree(image->iterated_points);
        cudaFree(image->squared_absolute_values);
        cudaFree(image->pixels_rgb);
        cudaFree(image->iterationsArr);
        cudaMallocManaged(&(image->points), total_pixels * sizeof(cuDoubleComplex));
        cudaMallocManaged(&(image->iterated_points), total_pixels * sizeof(cuDoubleComplex));
        cudaMallocManaged(&(image->squared_absolute_values), total_pixels * sizeof(double));
        cudaMallocManaged(&(image->pixels_rgb), total_pixels * 3 * sizeof(unsigned char));
        cudaMallocManaged(&(image->iterationsArr), total_pixels * sizeof(unsigned int));
    }
    else if (!(cuda_device_available)) {
        image->points = (cuDoubleComplex*)realloc(image->points, total_pixels * sizeof(cuDoubleComplex));
        image->iterated_points = (cuDoubleComplex*)realloc(image->iterated_points, total_pixels * sizeof(cuDoubleComplex));
        image->squared_absolute_values = (double*)realloc(image->squared_absolute_values, total_pixels * sizeof(double));
        image->pixels_rgb = (unsigned char*)realloc(image->pixels_rgb, total_pixels * 3 * sizeof(unsigned char));
        image->iterationsArr = (unsigned int*)realloc(image->iterationsArr, total_pixels * sizeof(unsigned int));
    }
}

void free_the_pointers(mandelbrot_image* image) {
    if (cuda_device_available) {
        cudaFree(image->points);
        cudaFree(image->iterated_points);
        cudaFree(image->squared_absolute_values);
        cudaFree(image->pixels_rgb);
        cudaFree(image->iterationsArr);
    }
    else if (!(cuda_device_available)) {
        free(image->points);
        free(image->iterated_points);
        free(image->squared_absolute_values);
        free(image->pixels_rgb);
        free(image->iterationsArr);
    }
}

void setup_incremental_iterations(mandelbrot_image* image) {
    if (incremental_iteration) {
        iterations_per_frame = incremental_iterations_per_frame;
    }
    else {
        iterations_per_frame = image->max_iterations;
    }
}

void draw_pixels(mandelbrot_image* image, GLFWwindow* window) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(image->resolution_x, image->resolution_y, GL_RGB, GL_UNSIGNED_BYTE, image->pixels_rgb);
    // Swap front and back buffers 
    glfwSwapBuffers(window);
}

int setup_glfw(GLFWwindow* window) {
    if (!window)
    {
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

void check_and_process_inputs(mandelbrot_image* image, GLFWwindow* window) {
    // Process keypresses if there were any:
    if (g_keypress_input_flag) {
        process_keyboard_input(g_last_keypress_input, image, window);
        reset_render_objects(image);
        g_keypress_input_flag = false;
    }
    else if (g_scroll_input_flag) {
        process_scroll_input(image, g_last_scroll_xoffset, g_last_scroll_yoffset);
        reset_render_objects(image);
        g_scroll_input_flag = false;
    }
    else if (g_resized_flag) {
        process_resize(image, window, g_resized_new_w, g_resized_new_h);
        reallocate_memory(image);
        reset_render_objects(image);
        g_resized_flag = false;
    }
}

void run_program_iteration(mandelbrot_image* image, GLFWwindow* window, char* window_title, int iterations_per_frame) {
    if (start_performance_test_flag) {
        g_application_mode = MODE_PERFORMANCE_TEST;
        start_performance_test(image);
    }
    if (g_application_mode == MODE_PERFORMANCE_TEST)
    {
        setup_performance_iteration(image);
        performance_iterations_done++;
        if (performance_iterations_done >= performance_iterations_total) {
            g_application_mode = MODE_VIEW;
            printf("\rPerformance test took %d ms.           \n", end_performance_test(image)); fflush(stdout);
        }
    }
    check_and_process_inputs(image, window);
    if ((g_application_mode == MODE_VIEW || g_application_mode == MODE_PERFORMANCE_TEST) && rendered_iterations < image->max_iterations) {
        mandelbrot_iterate_and_color(image);
        rendered_iterations += iterations_per_frame;
        // Rename the window title
        sprintf(window_title, "Drawing radius: %.32f | Max iterations: %d | center_re %.32f center_im: %.32f", image->draw_radius, image->max_iterations, image->center_real, image->center_imag);
        glfwSetWindowTitle(window, window_title);
        draw_pixels(image, window);
    }
}

int main() {
    mandelbrot_image* image;
    // TODO: move this to `allocate_memory`, but that would require a pointer to the pointer
    cudaMallocManaged(&image, sizeof(mandelbrot_image)); 
    setup_starting_parameters(image);
    printf("Done with starting params\n");

    // Check for CUDA devices:
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0)
    {
        int cuda_device_id;
        cudaGetDevice(&cuda_device_id);
        cudaDeviceProp cuda_device_properties;
        cudaGetDeviceProperties(&cuda_device_properties, cuda_device_id);
        cuda_device_available = true;
        printf("Using CUDA device: %s\n", cuda_device_properties.name);
        printf("cuda_num_blocks: %d\nblockSize: %d\n", cuda_num_blocks, cuda_block_size);
    } else {
        cuda_device_available = false;
        printf("No CUDA compatible devices found. Using CPU to compute images - performance will be limited.\n");
    }

    // Setup:
    setup_incremental_iterations(image);
    allocate_memory(image);
    build_complex_grid(image);
    mandelbrot_iterate_and_color(image);


    // Initialize the library 
    if (!glfwInit())
        return -1;
    // Create a windowed mode window and its OpenGL context 
    GLFWwindow* window = glfwCreateWindow(image->resolution_x, image->resolution_y, "Hello World", NULL, NULL);
    if (setup_glfw(window) == -1)
        return -1;
    char* window_title = (char*)malloc(1024);

    // Loop until the window is closed
    while (!glfwWindowShouldClose(window))
    {
        if (g_debugging_enabled)
        {
            Sleep(500); // cap fps to 2
        }
        run_program_iteration(image, window, window_title, iterations_per_frame);

        // Poll for and process events 
        glfwPollEvents();
    }

    glfwTerminate();
    free_the_pointers(image);

	return 0;
}
