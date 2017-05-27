#include <cuda.h>

/**
 * TODO: Document operation
 * TODO: Look for potential thrust implementation of this function
 */
__global__
void gpuBgrToLab(
        unsigned char* devBgr, 
        unsigned char* devL, unsigned char* devA, unsigned char* devB, 
        int rows, int cols, float gamma
        )
{
    // Compute pixel row, column indices
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = blockDim.y * blockIdx.y + threadIdx.y;


    if( row >= rows || col >= cols )
    {
        return;
    }

    // Threshold for RBG2Lab transform delta = (6 / 29) ** 3
    const float DELTA = 0.008856;
    // slope value is 1/(3 delta ** 2) and the intercept is 4/29
    const float DELTA_SLOPE = 7.787037;
    const float DELTA_INTERCEPT = 4.0 / 29.0;

    // Constants for LAB space
    const float L_MAX = 100.0;
    const float AB_MIN = -75.0;
    const float AB_MAX = 93.0;

    // Pixel index
    int c = row * cols + col;
    int idx = 3 * c;

    // Normalize to 0, 1 and apply gamma corrections
    float blue = (devBgr[idx]) / 255.0;
    float green = (devBgr[idx + 1]) / 255.0;
    float red = (devBgr[idx + 2]) / 255.0;

    blue = powf(blue, gamma);
    green = powf(green, gamma);
    red = powf(red, gamma);

    float x = (0.412453 * red) +  (0.357580 * green) + (0.180423 * blue);
    float y = (0.212671 * red) +  (0.715160 * green) + (0.072169 * blue);
    float z = (0.019334 * red) +  (0.119193 * green) + (0.950227 * blue);

    // Set white point = D65 (0.950456, 1.000000, 1.088754), 
    // y is not divided since it's coefficient is 1.000000
    // Read:
    // [1] https://en.wikipedia.org/wiki/Illuminant_D65
    // [2] https://en.wikipedia.org/wiki/White_point
    // [3] https://en.wikipedia.org/wiki/Lab_color_space#Forward_transformation
    x /= 0.950456;
    z /= 1.088754;
    
    // cbrt is the fast cube-root routine
    float fx = (x > DELTA) ? cbrtf(x) : (DELTA_SLOPE * x + DELTA_INTERCEPT);
    float fy = (y > DELTA) ? cbrtf(y) : (DELTA_SLOPE * y + DELTA_INTERCEPT);
    float fz = (z > DELTA) ? cbrtf(z) : (DELTA_SLOPE * z + DELTA_INTERCEPT);

    // CIEXYZ -> CIEL*a*b* transformation
    float l = (y > DELTA) ? (116 * cbrtf(y) - 16.0) : (903.3 * y);
    float a = 500.0 * (fx - fy);
    float b = 200.0 * (fy - fz);

    // Normalize L*, a*, b* values so they all lie in [0, 1]
    l /= L_MAX;
    a = (a - AB_MIN) / (AB_MAX - AB_MIN);
    b = (b - AB_MIN) / (AB_MAX - AB_MIN);
    l = (l < 0) ? 0 : ( l > 1 ? 1 : l);
    a = (a < 0) ? 0 : ( a > 1 ? 1 : a);
    b = (b < 0) ? 0 : ( b > 1 ? 1 : b);
    
    devL[c] = (unsigned char) (255 * l);
    devA[c] = (unsigned char) (255 * a);
    devB[c] = (unsigned char) (255 * b);
}


