#pragma once
#include <stdio.h>
#include <math.h>

typedef struct color_rgb {
	unsigned char r;
	unsigned char g;
	unsigned char b;
}color_rgb;

void set_color_rgb(color_rgb* color, unsigned char r, unsigned char g, unsigned char b);

color_rgb lerp_color(color_rgb start_color, color_rgb end_color, float factor);

// Some simple colors
color_rgb black;
color_rgb white;
color_rgb red;
color_rgb green;
color_rgb blue;
color_rgb blue_dark;

typedef struct simple_palette {
	color_rgb start_color;
	color_rgb end_color;
}simple_palette;

typedef struct palette {
	color_rgb colors[128]; // Can't think of a reason to have more than 128 colors in palette
	size_t length;
}palette;

palette palette_pretty;
palette palette_pastel;

void setup_palettes();
