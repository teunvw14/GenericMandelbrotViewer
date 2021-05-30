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
	color_rgb cp0 = { 66, 30, 15 };
	color_rgb cp1 = { 25, 7, 26 };
	color_rgb cp2 = { 9, 1, 47 };
	color_rgb cp3 = { 4, 4, 73 };
	color_rgb cp4 = { 0, 7, 100 };
	color_rgb cp5 = { 12, 44, 138 };
	color_rgb cp6 = { 24, 82, 177 };
	color_rgb cp7 = { 57, 125, 209 };
	color_rgb cp8 = { 134, 181, 229 };
	color_rgb cp9 = { 211, 236, 248 };
	color_rgb cp10 = { 241, 233, 191 };
	color_rgb cp11 = { 248, 201, 95 };
	color_rgb cp12 = { 255, 170, 0 };
	color_rgb cp13 = { 204, 128, 0 };
	color_rgb cp14 = { 153, 87, 0 };
	color_rgb cp15 = { 106, 52, 3 };
	palette_pretty.colors[0] = cp0;
	palette_pretty.colors[1] = cp1;
	palette_pretty.colors[2] = cp2;
	palette_pretty.colors[3] = cp3;
	palette_pretty.colors[4] = cp4;
	palette_pretty.colors[5] = cp5;
	palette_pretty.colors[6] = cp6;
	palette_pretty.colors[7] = cp7;
	palette_pretty.colors[8] = cp8;
	palette_pretty.colors[9] = cp9;
	palette_pretty.colors[10] = cp10;
	palette_pretty.colors[11] = cp11;
	palette_pretty.colors[12] = cp12;
	palette_pretty.colors[13] = cp13;
	palette_pretty.colors[14] = cp14;
	palette_pretty.colors[15] = cp15;

	// palette_pastel (very ugly)
	palette_pastel.length = 4;
	color_rgb cpl0 = { 44, 138, 97 };
	color_rgb cpl1 = { 186, 111, 214 };
	color_rgb cpl2 = { 90, 214, 60 };
	color_rgb cpl3 = { 214, 170, 78 };
	palette_pastel.colors[0] = cpl0;
	palette_pastel.colors[1] = cpl1;
	palette_pastel.colors[2] = cpl2;
	palette_pastel.colors[3] = cpl3;
}

color_rgb lerp_color(color_rgb start_color, color_rgb end_color, float factor) {
	// Linearly interpolate between two colors
	color_rgb result;
	if (factor >= 0 && factor <= 1) {
		result.r = start_color.r + (factor * (end_color.r - start_color.r));
		result.g = start_color.g + (factor * (end_color.g - start_color.g));
		result.b = start_color.b + (factor * (end_color.b - start_color.b));
	}
	return result;
}