#pragma once

#include <math.h>
#include <malloc.h>
#include <png.h>

void create_png(char* filename, int width, int height, png_bytep pixels_rgb);