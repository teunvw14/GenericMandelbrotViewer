#include <glfw3.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA imports
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "mandelbrot_image.h"
#include "starting_parameters.h"
#include "application.h"
#include "cuda_launch.h"


int main()
{
    mandelbrot_image* image;
    // TODO: move this to `allocate_memory`, but that would require a pointer to the pointer
    cudaMallocManaged(&image, sizeof(mandelbrot_image), cudaMemAttachGlobal);

    setup_starting_parameters(image);

    // Check for CUDA devices:
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        int cuda_device_id;
        cudaGetDevice(&cuda_device_id);
        struct cudaDeviceProp cuda_device_properties;
        cudaGetDeviceProperties(&cuda_device_properties, cuda_device_id);
        g_cuda_device_available = true;
        printf("Using CUDA device: %s\n", cuda_device_properties.name);
        printf("g_cuda_num_blocks: %d\nblockSize: %d\n", g_cuda_num_blocks, g_cuda_block_size);
    } else {
        g_cuda_device_available = false;
        printf("No CUDA compatible devices found. Using CPU to compute images - performance will be limited.\n");
    }

    // Setup:
    setup_incremental_iterations(image);
    allocate_memory(&image);
    reset_render_objects(image); // reset, or rather: initialize rendering objects

    // Initialize the library
    if (!glfwInit())
        return -1;
    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(image->resolution_x, image->resolution_y, "Hello World", NULL, NULL);
    if (setup_glfw(window) == -1)
        return -1;
    char* window_title = (char*)malloc(1024);

    // Loop until the window is closed
    while (!glfwWindowShouldClose(window)) {
        run_program_iteration(&image, image, window, window_title, g_iterations_per_frame);
        // Poll for and process events
        glfwPollEvents();
    }

    glfwTerminate();
    free_the_pointers(image);

    return 0;
}
