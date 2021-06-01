#include "color_palette.h"

void set_color_rgb(color_rgb* color, unsigned char r, unsigned char g, unsigned char b) {
	color->r = r;
	color->g = g;
	color->b = b;
}

void setup_palettes() {
	// Create the palettes. It's done in this
	// file because the definition of the palettes
	// are in this function.

	// pretty palette
	palette_pretty.length = 16;
	set_color_rgb(palette_pretty.colors + 0,  66, 30, 15);
	set_color_rgb(palette_pretty.colors + 1,  25, 7, 26);
	set_color_rgb(palette_pretty.colors + 2,  9, 1, 47);
	set_color_rgb(palette_pretty.colors + 3,  4, 4, 73);
	set_color_rgb(palette_pretty.colors + 4,  0, 7, 100);
	set_color_rgb(palette_pretty.colors + 5,  12, 44, 138);
	set_color_rgb(palette_pretty.colors + 6,  24, 82, 177);
	set_color_rgb(palette_pretty.colors + 7,  57, 125, 209);
	set_color_rgb(palette_pretty.colors + 8,  134, 181, 229);
	set_color_rgb(palette_pretty.colors + 9,  211, 236, 248);
	set_color_rgb(palette_pretty.colors + 10, 241, 233, 191);
	set_color_rgb(palette_pretty.colors + 11, 248, 201, 95);
	set_color_rgb(palette_pretty.colors + 12, 255, 170, 0);
	set_color_rgb(palette_pretty.colors + 13, 204, 128, 0);
	set_color_rgb(palette_pretty.colors + 14, 153, 87, 0);
	set_color_rgb(palette_pretty.colors + 15, 106, 52, 3);

	// palette_pastel (very ugly)
	palette_pastel.length = 4;
	set_color_rgb(palette_pastel.colors + 0, 44, 138, 97);
	set_color_rgb(palette_pastel.colors + 1, 186, 111, 214);
	set_color_rgb(palette_pastel.colors + 2, 90, 214, 60);
	set_color_rgb(palette_pastel.colors + 3, 214, 170, 78);
}

color_rgb lerp_color(color_rgb start_color, color_rgb end_color, float factor) {
	// Linearly interpolate between two colors
	color_rgb result;
	if (factor > 1) {
		//fprintf(stderr, "WARNING: Capping factor to 1 inside `lerp_color`. Something is wrong with the input.");
		factor = 1;
	}
	if (factor < 0) {
		//fprintf(stderr, "WARNING: Capping factor to 0 inside `lerp_color`. Something is wrong with the input.");
		factor = 0;
	}
	result.r = start_color.r + (factor * (end_color.r - start_color.r));
	result.g = start_color.g + (factor * (end_color.g - start_color.g));
	result.b = start_color.b + (factor * (end_color.b - start_color.b));
	return result;
}
