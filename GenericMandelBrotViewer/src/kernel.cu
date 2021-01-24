#include <glfw3.h>
#include <stdio.h>

// CUDA imports
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/complex.h>

// Debugging:
#include <windows.h>
bool debugging_enabled = false;

// Define starting parameters for the mandelbrot
double center_x = 0.0;
double center_y = 0.0;
int resolution_x = 1024;
int resolution_y = 1024;
double draw_radius = 2.5;
double escape_radius_squared = 4; // escape_radius = 2^7 = 256
int max_iterations = 64;

bool incremental_iteration = false;
int iterations_per_frame; // value set in main()
int incremental_iterations_per_frame = 4;
int rendered_iterations = 0;

// Cuda parameters:
int cuda_block_size = 256;
int cuda_num_blocks = int(ceil(resolution_x * resolution_y / cuda_block_size));
bool cuda_device_available = false;

// Define variables used to store values for each pixel
thrust::complex<double>* points;
thrust::complex<double>* iterated_points;
double* squared_absolute_values;
unsigned char* pixels_rgb;
unsigned short* iterationsArr; // TODO: Change to unsigned int. The program completely breaks if max_iterations >= 2^16 (= 65536) because of the unsigned short data type used here.


__global__ void build_complex_grid_cuda(
    double center_x, double center_y, 
    double draw_radius, 
    int resolution_x, int resolution_y, 
    thrust::complex<double>* points, 
    thrust::complex<double>* iterated_points) 
{
    // Create a grid of complex numbers around the center point (center_x, center_y).
    
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
		point_im = center_y + pixel_y * step_y - draw_radius;
		for (int pixel_x = 0; pixel_x < resolution_x; pixel_x += 1)
		{
			index = pixel_y * resolution_y + pixel_x;
			point_re = center_x + pixel_x * step_x - draw_radius;
			points[index] = thrust::complex<double>(point_re, point_im);
            iterated_points[index] = thrust::complex<double>(point_re, point_im);
		}
	}
}

void build_complex_grid_non_cuda(
        double center_x, double center_y,
        double draw_radius,
        int resolution_x, int resolution_y,
        thrust::complex<double>* points,
        thrust::complex<double>* iterated_points)
{
    // Create a grid of complex numbers around the center point (center_x, center_y).
    double step_y = 2 * draw_radius / resolution_y;
    double step_x = 2 * draw_radius / resolution_x;
    double point_re;
    double point_im;
    int index;
    // Start drawing in the bottom left, go row by row.
    for (int pixel_y = 1; pixel_y < resolution_y; pixel_y++)
    {
        point_im = center_y + pixel_y * step_y - draw_radius;
        for (int pixel_x = 0; pixel_x < resolution_x; pixel_x++)
        {
            index = pixel_y * resolution_y + pixel_x;
            point_re = center_x + pixel_x * step_x - draw_radius;
            points[index] = thrust::complex<double>(point_re, point_im);
            iterated_points[index] = thrust::complex<double>(point_re, point_im);
        }
    }
}

__global__ void mandelbrot_iterate_cuda(
    int max_iterations,
    double escape_radius_squared,
    int resolution_x, int resolution_y,
    thrust::complex<double>* points,
    thrust::complex<double>* iterated_points,
    double* squared_absolute_values,
    unsigned short* iterationsArr
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
            thrust::complex<double> c = points[index];
            thrust::complex<double> it_point = iterated_points[index];
            double sq_abs = squared_absolute_values[index];
            unsigned short iterations_ = iterationsArr[index];
            while (iterations_ < max_iterations && sq_abs < escape_radius_squared) {
                it_point = thrust::complex<double>(it_point.real() * it_point.real() - it_point.imag() * it_point.imag(), 2 * it_point.real() * it_point.imag()) + c; // z^2 + c
                sq_abs = it_point.real() * it_point.real() + it_point.imag() * it_point.imag();
                iterations_++;
            }
            iterated_points[index] = it_point;
            iterationsArr[index] = iterations_;
            squared_absolute_values[index] = sq_abs;
        }
    }
}

void mandelbrot_iterate_non_cuda(
    int max_iterations,
    double escape_radius_squared,
    int resolution_x, int resolution_y,
    thrust::complex<double>* points,
    thrust::complex<double>* iterated_points,
    double* squared_absolute_values,
    unsigned short* iterationsArr
)
{
    int index;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = 1; pixel_y < resolution_y; pixel_y++)
    {
        for (int pixel_x = 1; pixel_x < resolution_x; pixel_x++)
        {
            // Calculate the iterations required for a given point to exceed the escape radius.
            index = pixel_y * resolution_y + pixel_x;
            thrust::complex<double> c = points[index];
            thrust::complex<double> it_point = iterated_points[index];
            double sq_abs = squared_absolute_values[index];
            unsigned short iterations_ = iterationsArr[index];
            while (iterations_ < max_iterations && sq_abs < escape_radius_squared) {
                it_point = thrust::complex<double>(it_point.real() * it_point.real() - it_point.imag() * it_point.imag(), 2 * it_point.real() * it_point.imag()) + c; // z^2 + c
                sq_abs = it_point.real() * it_point.real() + it_point.imag() * it_point.imag();
                iterations_++;
            }
            iterated_points[index] = it_point;
            iterationsArr[index] = iterations_;
            squared_absolute_values[index] = sq_abs;
        }
    }
}


__global__ void color_cuda(
    int max_iterations,
    unsigned short* iterationsArr,
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
    int iterations;

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
                    r = 1-m;
                    g = 1-m;
                    b = 1-m;
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


void color_non_cuda(
    int max_iterations,
    unsigned short* iterationsArr,
    double* squared_absolute_values,
    int resolution_x,
    int resolution_y,
    double draw_radius,
    unsigned char* rgb_data
)
{
    // Do some coloring!
    int index;
    int iterations;

    //printf("thread_index_x: %i | block_index_x: %i | thread_stride_x: %i | block_stride_x: %i\n", thread_index_x, block_index_x, thread_stride_x, block_stride_x);
    for (int pixel_y = 1; pixel_y < resolution_y; pixel_y++)
    {
        for (int pixel_x = 1; pixel_x < resolution_x; pixel_x++)
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

// GLFW
GLFWwindow* window;

void build_complex_grid()
{
    if (cuda_device_available) {
        build_complex_grid_cuda <<< 1, 1024 >>> (center_x, center_y, draw_radius, resolution_x, resolution_y, points, iterated_points);
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)){
        build_complex_grid_non_cuda(center_x, center_y, draw_radius, resolution_x, resolution_y, points, iterated_points);
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

    // Rebuild the grid of complex numbers based on (new) center and (new) center.
    build_complex_grid(); 
    // Reset the `squared_absolute_values` to zero by allocating the memory space again.
    if (cuda_device_available) {
        cudaFree(squared_absolute_values);
        cudaFree(iterationsArr);
        cudaMallocManaged(&squared_absolute_values, resolution_x * resolution_y * sizeof(double));
        cudaMallocManaged(&iterationsArr, resolution_x * resolution_y * sizeof(unsigned short));
        // Synchronize the GPU so the whole thing doesn't crash.
        cudaDeviceSynchronize();
    }
    else if (!(cuda_device_available)) {
        free(squared_absolute_values);
        free(iterationsArr);
        squared_absolute_values = (double*)malloc(resolution_x * resolution_y * sizeof(double));
        iterationsArr = (unsigned short*)malloc(resolution_x * resolution_y * sizeof(unsigned short));
    }

    // Reset the amount of rendered iterations to 0. 
    rendered_iterations = 0;
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
            //mandelbrot_iterate_and_color();
            break;
        case GLFW_KEY_MINUS: 
            draw_radius /= 0.75; // zoom out
            reset_render_objects();
            //mandelbrot_iterate_and_color();
            break;
        case GLFW_KEY_LEFT:
            center_x -= 0.1 * draw_radius;
            reset_render_objects();
            //mandelbrot_iterate_and_color();
            break;
        case GLFW_KEY_RIGHT:
            center_x += 0.1 * draw_radius;
            reset_render_objects();
            //mandelbrot_iterate_and_color();
            break;
        case GLFW_KEY_UP:
            center_y += 0.1 * draw_radius;
            reset_render_objects();
            //mandelbrot_iterate_and_color();
            break;
        case GLFW_KEY_DOWN:
            center_y -= 0.1 * draw_radius;
            reset_render_objects();
            //mandelbrot_iterate_and_color();
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
        cudaMallocManaged(&points, resolution_x * resolution_y * sizeof(thrust::complex<double>));
        cudaMallocManaged(&iterated_points, resolution_x * resolution_y * sizeof(thrust::complex<double>));
        cudaMallocManaged(&squared_absolute_values, resolution_x * resolution_y * sizeof(double));
        cudaMallocManaged(&pixels_rgb, resolution_x * resolution_y * 3 * sizeof(unsigned char));
        cudaMallocManaged(&iterationsArr, resolution_x * resolution_y * sizeof(unsigned short));
    }
    else if (!(cuda_device_available)) {
        points = (thrust::complex<double>*)malloc(resolution_x * resolution_y * sizeof(thrust::complex<double>));
        iterated_points = (thrust::complex<double>*)malloc(resolution_x * resolution_y * sizeof(thrust::complex<double>));
        squared_absolute_values = (double*)malloc(resolution_x * resolution_y * sizeof(double));
        pixels_rgb = (unsigned char*)malloc(resolution_x * resolution_y * 3 * sizeof(unsigned char));
        iterationsArr = (unsigned short*)malloc(resolution_x * resolution_y * sizeof(unsigned short));
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
    window = glfwCreateWindow(resolution_x, resolution_y, "Hello World", NULL, NULL);
    char* window_title = (char*)malloc(256);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // Loop until the window is closed
    while (!glfwWindowShouldClose(window))
    {
        if (debugging_enabled)
        {
            Sleep(500); // set max fps to 2
        }

        // Render here 
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        // TODO: make it so that the iterations increase for as long as the center and draw_radius are the same - up to the max of course
        if (rendered_iterations < max_iterations) {
            printf("rendering %d iterations\n", iterations_per_frame);
            mandelbrot_iterate_n_and_color(iterations_per_frame);
            rendered_iterations += iterations_per_frame;
            sprintf(window_title, "Max iterations: %d | points[0]: RE: %.32f IM: %.32f", max_iterations, points[0].real(), points[0].imag());
            glfwSetWindowTitle(window, window_title);
        }

        glDrawPixels(resolution_x, resolution_y, GL_RGB, GL_UNSIGNED_BYTE, pixels_rgb);

        // Swap front and back buffers 
        glfwSwapBuffers(window);

        // Poll for and process events 
        glfwPollEvents();
    }

    glfwTerminate();
    free_the_pointers();

	return 0;
}
