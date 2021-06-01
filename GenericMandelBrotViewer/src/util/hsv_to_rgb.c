#include "color_palette.h"

color_rgb hsv_to_rgb(float H, float S, float V) {
    color_rgb result;
    // HSV to RGB conversion, yay!
    // TODO: look into edge cases for H and why they happen.
    //if (H > 360 || H < 0 || S > 1 || S < 0 || V > 1 || V < 0)
    //{
    //printf("The given HSV values are not in valid range.\n H: %f S: %.2f, V: %.2f\n", H, S, V);
    //printf("Iterations: %f\n", f_iterations);
    //}
    float h = H / 60;
    float C = S * V;
    float X = C * (1 - fabsf((fmodf(h, 2) - 1)));
    float m = V - C;
    float r, g, b;
    if (h >= 0 && h <= 1) {
        r = C;
        g = X;
        b = 0;
    }
    else if (h > 1 && h < 2) {
        r = X;
        g = C;
        b = 0;
    }
    else if (h > 2 && h <= 3) {
        r = 0;
        g = C;
        b = X;
    }
    else if (h > 3 && h <= 4) {
        r = 0;
        g = X;
        b = C;
    }
    else if (h > 4 && h <= 5) {
        r = X;
        g = 0;
        b = C;
    }
    else if (h > 5 && h <= 6) {
        r = C;
        g = 0;
        b = X;
    }
    else {
        r = 1 - m;
        g = 1 - m;
        b = 1 - m;
    }
    result.r = (r + m) * 255;
    result.g = (g + m) * 255;
    result.b = (b + m) * 255;
    return result;
}
