#include <math.h>
#include <malloc.h>
#include <png.h>

void create_png(char* filename, int width, int height, png_bytep pixels_rgb)
{

	FILE* f = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;
	int row_length = width * 3 * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_length);

	// Open file
	f = fopen(filename, "wb");
	if (f == NULL) {
		fprintf(stderr, "Something went wrong opening file %s\n", filename);
		goto finalise;
	}

	// Initialize write structure
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if (png_ptr == NULL) {
		fprintf(stderr, "Something went wrong creating png struct...\n");
		goto finalise;
	}

	// Initialize info structure
	info_ptr = png_create_info_struct(png_ptr);
	if (info_ptr == NULL) {
		fprintf(stderr, "Something went wrong creating png info struct...\n");
		goto finalise;
	}

	png_init_io(png_ptr, f);

	// Write header (8 bit colour depth)
	png_set_IHDR(png_ptr, info_ptr, width, height,
		8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	png_write_info(png_ptr, info_ptr);

	// Write the image data row by row
	// For some reason this only works when y is "reversed"
	// TODO: figure out why
	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			int offset = y * row_length + x * 3;
			row[x * 3 + 0] = pixels_rgb[offset + 0];
			row[x * 3 + 1] = pixels_rgb[offset + 1];
			row[x * 3 + 2] = pixels_rgb[offset + 2];
		}
		png_write_row(png_ptr, row);
	}

	// Tell libpng we're done with the image
	png_write_end(png_ptr, NULL);

finalise:
	if (f != NULL) fclose(f);
	if (info_ptr != NULL) png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	if (png_ptr != NULL) png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	if (row != NULL) free(row);
}
