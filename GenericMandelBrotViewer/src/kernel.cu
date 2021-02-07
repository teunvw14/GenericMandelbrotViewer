#include <glfw3.h>
#include <stdio.h>
#include <math.h>

// CUDA imports
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <device_launch_parameters.h>

// Debugging:
#include <sys/timeb.h>
#include <windows.h>
bool debugging_enabled = false;
struct timeb start, end;
int performance_iterations_total = 512;
int performance_iterations_done;
bool start_performance_test_flag = false;
int max_iterations_store;
int center_real_store;
int center_imag_store;
double draw_radius_store;

// Define starting parameters for the mandelbrot
double center_real = 0.0;
double center_imag = 0.0;
int resolution_x = 1024;
int resolution_y = 1024;
double draw_radius = 2.5;
double escape_radius_squared = 4; // escape_radius = 2^7 = 256
int max_iterations = 64;

#define MODE_VIEW 0
#define MODE_PERFORMANCE_TEST 1
unsigned short application_mode = MODE_VIEW;
bool incremental_iteration = false;
int iterations_per_frame; // value set in main()
int incremental_iterations_per_frame = 4;
int rendered_iterations = 0;

// Cuda parameters:
int cuda_block_size = 256;
int cuda_num_blocks = int(ceil(resolution_x * resolution_y / cuda_block_size));
bool cuda_device_available = false;

// Define variables used to imaginary number values for each pixel
cuDoubleComplex* points;
cuDoubleComplex* iterated_points;
double* squared_absolute_values;
unsigned char* pixels_rgb;
unsigned int* iterationsArr;

__global__ void build_complex_grid_cuda(
    double center_real, double center_imag, 
    double draw_radius, 
    int resolution_x, int resolution_y, 
    cuDoubleComplex* points,
    cuDoubleComplex* iterated_points
    )
{
    // Create a grid of complex numbers around the center point (center_real, center_imag).
    
    int thread_index = threadIdx.x;
	int thread_stride = blockDim.x;

	double step_y = 2 * draw_radius / resolution_y;
	double step_x = 2 * draw_radius / resolution_x;
	double point_re;
	double point_im;
	int index;
	// Start drawing in the bottom left, go row by row.
	for (int pixel_y = thread_index; pixel_y < resolution_y; pixel_y += thread_stride)
	{
		point_im = center_imag + pixel_y * step_y - draw_radius;
		for (int pixel_x = 0; pixel_x < resolution_x; pixel_x++)
		{
			index = pixel_y * resolution_y + pixel_x;
			point_re = center_real + pixel_x * step_x - draw_radius;
            points[index] = make_cuDoubleComplex(point_re, point_im);
            iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
		}
	}
}

void build_complex_grid_non_cuda(
        double center_real, double center_imag,
        double draw_radius,
        int resolution_x, int resolution_y,
        cuDoubleComplex* points,
        cuDoubleComplex* iterated_points
    )
{
    // Create a grid of complex numbers around the center point (center_real, center_imag).
    double step_y = 2 * draw_radius / resolution_y;
    double step_x = 2 * draw_radius / resolution_x;
    double point_re;
    double point_im;
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 0; pixel_y < resolution_y; pixel_y++)
    {
        point_im = center_imag + pixel_y * step_y - draw_radius;
        for (int pixel_x = 0; pixel_x < resolution_x; pixel_x++)
        {
            index = pixel_y * resolution_y + pixel_x;
            point_re = center_real + pixel_x * step_x - draw_radius;
            points[index] = make_cuDoubleComplex(point_re, point_im);
            iterated_points[index] = make_cuDoubleComplex(point_re, point_im);
        }
    }
}

__global__ void mandelbrot_iterate_cuda(
    int max_iterations,
    double escape_radius_squared,
    int resolution_x, int resolution_y,
    cuDoubleComplex* points,
    cuDoubleComplex* iterated_points,
    double* squared_absolute_values,
    unsigned int* iterationsArr
)
{
    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;
    //int block_index_y = blockIdx.y;
    //int block_stride_y = gridDim.y;

    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;
    //int thread_index_y = threadIdx.y;
    //int thread_stride_y = blockDim.y;
    int index;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = block_index_x; pixel_y < resolution_y; pixel_y += block_stride_x)
    {
        for (int pixel_x = thread_index_x; pixel_x < resolution_x; pixel_x += thread_stride_x)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * resolution_y + pixel_x;
            cuDoubleComplex starting_number = points[index];
            cuDoubleComplex iterated_point = iterated_points[index];
            double sq_abs = squared_absolute_values[index];
            unsigned int iterations_ = iterationsArr[index];
            while (iterations_ < max_iterations && sq_abs < escape_radius_squared) {
                iterated_point = make_cuDoubleComplex(iterated_point.x * iterated_point.x - iterated_point.y * iterated_point.y + starting_number.x,
                    2 * iterated_point.x * iterated_point.y + starting_number.y);
                sq_abs = iterated_point.x * iterated_point.x + iterated_point.y * iterated_point.y;
                iterations_++;
            }
            iterated_points[index] = iterated_point;
            iterationsArr[index] = iterations_;
            squared_absolute_values[index] = sq_abs;
        }
    }
}

void mandelbrot_iterate_non_cuda(
    int max_iterations,
    double escape_radius_squared,
    int resolution_x, int resolution_y,
    cuDoubleComplex* points,
    cuDoubleComplex* iterated_points,
    double* squared_absolute_values,
    unsigned int* iterationsArr
)
{
    int index = 0;

    for (int pixel_y = 0; pixel_y < resolution_y; pixel_y++)
    {
        for (int pixel_x = 0; pixel_x < resolution_x; pixel_x++)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * resolution_y + pixel_x;
            cuDoubleComplex starting_number = points[index];
            cuDoubleComplex iterated_point = iterated_points[index];
            double sq_abs = squared_absolute_values[index];
            unsigned int iterations_ = iterationsArr[index];
            while (iterations_ < max_iterations && sq_abs < escape_radius_squared) {
                iterated_point = make_cuDoubleComplex(iterated_point.x * iterated_point.x - iterated_point.y * iterated_point.y + starting_number.x,
                    2 * iterated_point.x * iterated_point.y + starting_number.y);
                sq_abs = iterated_point.x * iterated_point.x + iterated_point.y * iterated_point.y;
                iterations_++;
            }
            iterated_points[index] = iterated_point;
            iterationsArr[index] = iterations_;
            squared_absolute_values[index] = sq_abs;
        }
    }
}


__global__ void color_cuda(
    int max_iterations,
    unsigned int* iterationsArr,
    double * squared_absolute_values,
    int resolution_x,
    int resolution_y,
    double draw_radius,
    unsigned char * rgb_data
)
{
    // Do some coloring!

    int block_index_x = blockIdx.x;
    int block_stride_x = gridDim.x;
    //int block_index_y = blockIdx.y;
    //int block_stride_y = gridDim.y;

    int thread_index_x = threadIdx.x;
    int thread_stride_x = blockDim.x;
    //int thread_index_y = threadIdx.y;
    //int thread_stride_y = blockDim.y;
    int index;
    unsigned int iterations;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = block_index_x; pixel_y < resolution_y; pixel_y += block_stride_x)
    {
        for (int pixel_x = thread_index_x; pixel_x < resolution_x; pixel_x += thread_stride_x)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * resolution_y + pixel_x;
            iterations = iterationsArr[index];
            if (iterations == max_iterations)
            {
                // Values that don't escape are colored black:
                rgb_data[3 * index + 0] = 25; // Red value
                rgb_data[3 * index + 1] = 25; // Green value
                rgb_data[3 * index + 2] = 25; // Blue value
            } 
            else
            {
                float f_iterations = (float)iterations;
                float f_max_iterations = (float)max_iterations;
                // Smooth colors!
                float escape_size = __double2float_rn(squared_absolute_values[index]);
                float smoothed_iterations = iterations + 1 - log2f(log(escape_size)) + sqrtf(sqrtf(draw_radius));
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

                rgb_data[3 * index + 0] = red; // Red value
                rgb_data[3 * index + 1] = green; // Green value
                rgb_data[3 * index + 2] = blue; // Blue value
            }
        }
    }
}


void color_non_cuda(
    int max_iterations,
    unsigned int* iterationsArr,
    double* squared_absolute_values,
    int resolution_x,
    int resolution_y,
    double draw_radius,
    unsigned char* rgb_data
)
{
    // Do some coloring!
    int index;
    unsigned int iterations;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = 0; pixel_y < resolution_y; pixel_y++)
    {
        for (int pixel_x = 0; pixel_x < resolution_x; pixel_x++)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * resolution_y + pixel_x;
            iterations = iterationsArr[index];
            if (iterations == max_iterations)
            {
                // Values that don't escape are colored black:
                rgb_data[3 * index + 0] = 25; // Red value
                rgb_data[3 * index + 1] = 25; // Green value
                rgb_data[3 * index + 2] = 25; // Blue value
            }
            else
            {
                float f_iterations = (float)iterations;
                float f_max_iterations = (float)max_iterations;
                // Smooth colors!
                float escape_size = (float )(squared_absolute_values[index]);
                float smoothed_iterations = iterations + 1 - log2f(log(escape_size)) + sqrtf(sqrtf(draw_radius));
                float H = 360 * smoothed_iterations / f_max_iterations;
                float S = .65;
                float V = 1;


#pragma region HSV_to_RGB_Conversion
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
#pragma endregion

                // Cap RGB values to 255
                if (red > 255) { red = 255; }
                if (green > 255) { green = 255; }
                if (blue > 255) { blue = 255; }

                rgb_data[3 * index + 0] = red; // Red value
                rgb_data[3 * index + 1] = green; // Green value
                rgb_data[3 * index + 2] = blue; // Blue value
            }
        }
    }
}

void build_complex_grid()
{
    if (cuda_device_available) {
        build_complex_grid_cuda <<< 1, 1024 >>> (center_real, center_imag, draw_radius, resolution_x, resolution_y, points, iterated_points);
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)){
        build_complex_grid_non_cuda(center_real, center_imag, draw_radius, resolution_x, resolution_y, points, iterated_points);
    }
}

void mandelbrot_iterate_and_color()
{
    if (cuda_device_available) {
        mandelbrot_iterate_cuda <<< cuda_num_blocks, cuda_block_size
            >>> (
                max_iterations,
                escape_radius_squared,
                resolution_x, resolution_y,
                points,
                iterated_points,
                squared_absolute_values,
                iterationsArr
            );

        cudaDeviceSynchronize();
        color_cuda <<< cuda_num_blocks, cuda_block_size
        >>> (
                max_iterations,
                iterationsArr,
                squared_absolute_values,
                resolution_x,
                resolution_y,
                draw_radius,
                pixels_rgb
            );
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)) {
        mandelbrot_iterate_non_cuda(
                max_iterations,
                escape_radius_squared,
                resolution_x, resolution_y,
                points,
                iterated_points,
                squared_absolute_values,
                iterationsArr
            );

        color_non_cuda(
                max_iterations,
                iterationsArr,
                squared_absolute_values,
                resolution_x,
                resolution_y,
                draw_radius,
                pixels_rgb
            );
    }
}

// Under maintenance.
void mandelbrot_iterate_n_and_color(int iterations)
{
    mandelbrot_iterate_and_color();
    //mandelbrot_iterate_and_color_cuda << < cuda_num_blocks, cuda_block_size >> > (iterations, escape_radius_squared, resolution_x, resolution_y, points, iterated_points, squared_absolute_values, pixels_rgb);
}


void reset_render_objects()
{
    // This function resets all the variables that are used for rendering the Mandelbrot. 


    // Reset the `squared_absolute_values` to zero by allocating the memory space again.
    if (cuda_device_available) {
        cudaFree(squared_absolute_values);
        cudaFree(iterationsArr);
        cudaMallocManaged(&squared_absolute_values, resolution_x * resolution_y * sizeof(double));
        cudaMallocManaged(&iterationsArr, resolution_x * resolution_y * sizeof(unsigned int));
        // Synchronize the GPU so the whole thing doesn't crash.
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)) {
        free(squared_absolute_values);
        free(iterationsArr);
        squared_absolute_values = (double*)malloc(resolution_x * resolution_y * sizeof(double));
        iterationsArr = (unsigned int*)malloc(resolution_x * resolution_y * sizeof(unsigned int));
    }
    // Rebuild the grid of complex numbers based on (new) center_real and (new) center_imag.
    build_complex_grid();

    // Reset the amount of rendered iterations to 0. 
    rendered_iterations = 0;
}



// Performance testing functions below:

void start_performance_test() {
    printf("Starting performance test.\n");
    start_performance_test_flag = false;
    performance_iterations_done = 0;
    rendered_iterations = 0;
    reset_render_objects();
    ftime(&start);

    // Set parameters for test:
    max_iterations_store = max_iterations;
    center_real_store = center_real;
    center_imag_store = center_imag;
    draw_radius_store = draw_radius;
}

void setup_performance_iteration() {

    int starting_max_iterations;
    // Choose a spot to move to, and change starting max_iterations accordingly
    unsigned short spot = ceil(4.0 * (float) performance_iterations_done / (float) performance_iterations_total);
    if (spot <= 0) { spot = 1; }
    switch (spot) {
    case 1:
        center_real = -1.769249938555972345710642912303869;
        center_imag = -0.05694208981877081632294590463061;
        draw_radius = 1.9 * pow(10, -13);
        starting_max_iterations = 512;
        break;
    case 2:
        center_real = -0.0452407411;
        center_imag = 0.9868162204352258;
        draw_radius = 4.4 * pow(10, -9);
        starting_max_iterations = 200;
        break;
    case 3:
        center_real = -0.7336438924199521;
        center_imag = 0.2455211406714035;
        draw_radius = 4.5 * pow(10, -14);
        starting_max_iterations = 624;
        break;
    case 4:
        center_real = -0.0452407411;
        center_imag = 0.9868162204352258;
        draw_radius = 4.4 * pow(10, -9);
        starting_max_iterations = 256;
        break;
    default:
        break;
    }
    // hack to start iterating from `starting_max_iterations` for each new spot
    max_iterations = starting_max_iterations + performance_iterations_done * 4 - ((performance_iterations_total - 1) * 4 * (spot - 1)/ 4);
    printf("\rRendering spot %d with %d iterations.", spot, max_iterations); fflush(stdout);

    reset_render_objects();
}

int end_performance_test() {
    ftime(&end);
    max_iterations = max_iterations_store;
    center_real = center_real_store;
    center_imag = center_imag_store;
    draw_radius = draw_radius_store;
    reset_render_objects();
    int elapsed_time = (int)1000.0 * (end.time - start.time) + (end.millitm - start.millitm);
    return elapsed_time;
}


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key) {
        case GLFW_KEY_D:
            if (debugging_enabled) {
                debugging_enabled = false;
            }
            else {
                debugging_enabled = true;
            }
            reset_render_objects();
            break;
        case GLFW_KEY_EQUAL: // zoom in, = is also +
            draw_radius *= 0.75; // zoom in
            reset_render_objects();
            break;
        case GLFW_KEY_MINUS: 
            draw_radius /= 0.75; // zoom out
            reset_render_objects();
            break;
        case GLFW_KEY_LEFT:
                center_real -= 0.1 * draw_radius;
            printf("%f\n", center_real);
            reset_render_objects();
            break;
        case GLFW_KEY_RIGHT:
            center_real += 0.1 * draw_radius;
            reset_render_objects();
            printf("%f\n", center_real);
            break;
        case GLFW_KEY_UP:
            center_imag += 0.1 * draw_radius;
            reset_render_objects();
            break;
        case GLFW_KEY_DOWN:
            center_imag -= 0.1 * draw_radius;
            reset_render_objects();
            break;
        case GLFW_KEY_LEFT_BRACKET:
            if (max_iterations > 2 && max_iterations < 10) {
                max_iterations--;
            }
            else if (max_iterations >= 10) {
                max_iterations *= 0.9;
            }
            printf("Max iterations now at: %d\n", max_iterations);
            if (incremental_iteration) {
                iterations_per_frame = incremental_iterations_per_frame;
            }
            else {
                iterations_per_frame = max_iterations;
            }
            reset_render_objects();
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            if (max_iterations < 10) {
                max_iterations++;
            }
            else if (max_iterations >= 10){
                max_iterations /= 0.9;
            }
            printf("Max iterations now at: %d\n", max_iterations);
            if (incremental_iteration) {
                iterations_per_frame = incremental_iterations_per_frame;
            }
            else {
                iterations_per_frame = max_iterations;
            }
            reset_render_objects();
            break;
        case GLFW_KEY_I:
            if (incremental_iteration)
            {
                iterations_per_frame = incremental_iterations_per_frame;
                incremental_iteration = false;
            }
            else {
                iterations_per_frame = max_iterations;
                incremental_iteration = true;
            }
            break;
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
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if (yoffset > 0) {
        draw_radius *= 0.75; // zoom in
        reset_render_objects();
    }
    else if (yoffset < 0) {
        draw_radius /= 0.75; // zoom out
        reset_render_objects();
    }
}

void allocate_memory() {
    if (cuda_device_available) {
        cudaMallocManaged(&points, resolution_x * resolution_y * sizeof(cuDoubleComplex));
        cudaMallocManaged(&iterated_points, resolution_x * resolution_y * sizeof(cuDoubleComplex));
        cudaMallocManaged(&squared_absolute_values, resolution_x * resolution_y * sizeof(double));
        cudaMallocManaged(&pixels_rgb, resolution_x * resolution_y * 3 * sizeof(unsigned char));
        cudaMallocManaged(&iterationsArr, resolution_x * resolution_y * sizeof(unsigned int));
    }
    else if (!(cuda_device_available)) {
        points = (cuDoubleComplex *)malloc(resolution_x * resolution_y * sizeof(cuDoubleComplex));
        iterated_points = (cuDoubleComplex *)malloc(resolution_x * resolution_y * sizeof(cuDoubleComplex));
        squared_absolute_values = (double*)malloc(resolution_x * resolution_y * sizeof(double));
        pixels_rgb = (unsigned char*)malloc(resolution_x * resolution_y * 3 * sizeof(unsigned char));
        iterationsArr = (unsigned int*)malloc(resolution_x * resolution_y * sizeof(unsigned int));
    }
}

void free_the_pointers() {
    if (cuda_device_available) {
        cudaFree(points);
        cudaFree(iterated_points);
        cudaFree(squared_absolute_values);
        cudaFree(pixels_rgb);
        cudaFree(iterationsArr);
    }
    else if (!(cuda_device_available)) {
        free(points);
        free(iterated_points);
        free(squared_absolute_values);
        free(pixels_rgb);
        free(iterationsArr);
    }
}

void setup_incremental_iterations() {
    if (incremental_iteration) {
        iterations_per_frame = incremental_iterations_per_frame;
    }
    else {
        iterations_per_frame = max_iterations;
    }
}

void draw_pixels(GLFWwindow* window) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawPixels(resolution_x, resolution_y, GL_RGB, GL_UNSIGNED_BYTE, pixels_rgb);

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
}


void run_program_iteration(GLFWwindow* window, char* window_title, int iterations_per_frame) {
    if (start_performance_test_flag) {
        application_mode = MODE_PERFORMANCE_TEST;
        start_performance_test();
    }
    if (application_mode == MODE_PERFORMANCE_TEST)
    {
        setup_performance_iteration();
        performance_iterations_done++;
        if (performance_iterations_done >= performance_iterations_total) {
            application_mode = MODE_VIEW;
            printf("\rPerformance test took %d ms.           \n", end_performance_test()); fflush(stdout);
        }
    }
    if ((application_mode == MODE_VIEW || application_mode == MODE_PERFORMANCE_TEST) && rendered_iterations < max_iterations) {
        mandelbrot_iterate_n_and_color(iterations_per_frame);
        rendered_iterations += iterations_per_frame;
        // Rename the window title
        sprintf(window_title, "Drawing radius: %.32f | Max iterations: %d | center_re %.32f center_im: %.32f", draw_radius, max_iterations, center_real, center_imag);
        glfwSetWindowTitle(window, window_title);
    }
}


int main() {
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
    allocate_memory();
    build_complex_grid();
    mandelbrot_iterate_and_color();
    setup_incremental_iterations();

    // Initialize the library 
    if (!glfwInit())
        return -1;
    // Create a windowed mode window and its OpenGL context 
    GLFWwindow* window = glfwCreateWindow(resolution_x, resolution_y, "Hello World", NULL, NULL);
    if (!setup_glfw(window))
        return -1;
    char* window_title = (char*)malloc(1024);

    // Loop until the window is closed
    while (!glfwWindowShouldClose(window))
    {
        if (debugging_enabled)
        {
            Sleep(500); // cap fps to 2
        }
        run_program_iteration(window, window_title, iterations_per_frame);
        
        draw_pixels(window);
        // Poll for and process events 
        glfwPollEvents();
    }

    glfwTerminate();
    free_the_pointers();

	return 0;
}
