# Generic Mandelbrot Viewer

This project is a CUDA/C implementation of a Mandelbrot Set viewer. If you're not familiar with the Mandelbrot Set: it's a beautiful mathematical object (just look at the [samples](#samples)). A passion project. 

# Background info

I've had a fascination for the Mandelbrot Set for a while now - which is why I created the [pybrot project](https://github.com/teunvw14/pybrot) using Python during my 2018 High School summer break; I learned a bunch and was pretty happy with the end result. 

Unfortunately, because it was written in Python, pybrot wasn't particularly fast. For that reason, I've wanted to implement a mandelbrot viewer/renderer in C ever since I finished the original Python project. 

And so this project was born. I wanted to see how much I could improve on performance if I chose a programming language that was closer to the metal. 

Hence why I created this project: a Mandelbrot viewer written in C / CUDA. The results were satisfactory: this implementation is roughly **five hundred times faster** than my pybrot implementation. Whoo! 

This project taught me:
- the basics of GPU computing using CUDA; and
- how to actually use the fantastic C programming language; and
- most importantly: how pointers work.

# Samples


[<img src="samples/starting_view.png" width="512"/>](samples/starting_view.png)

> The classic Mandelbrot Set image.

[<img src="samples/mandel_wave.png" width="512"/>](samples/mandel_wave.png)

> The crest between two of the Mandelbrot's "Bulbs".

[<img src="samples/mandelbrot_whirlpool.gif" width="512"/>](samples/mandelbrot_whirlpool.gif)

> Don't stare at this one too long. 

[<img src="samples/minibrot.png" width="512"/>](samples/minibrot.png)

> One of the Manelbrot's many self-contained duplicates.

[<img src="samples/mandel_organs.png" width="512"/>](samples/mandel_organs.png)

> Beautiful, isn't it? 


# Features

- Basic controls for viewing the mandelbrot set
- Speedy performance thanks to NVIDIA GPU acceleration. Also works on systems without an NVIDIA GPU. 
- Performance test to check performance of (future) optimisations.

# Controls

- Pan image: arrow keys (up, down, right, left)
- Zoom in/out: `+` and `-` keys respectively
- Increase maximum iterations: `[`
- Do a performance test and print results to console: `e` key
- Quit: `esc`

# Build

### Requirements
To build this projects: the following software is required. 
- Visual Studio 19
- The [CUDA Toolkit](developer.NVIDIA.com/cuda-downloads)

### Build using Visual Studio 19
Open the solution in Visual Studio, and build. (`Build > Build Solution`)
