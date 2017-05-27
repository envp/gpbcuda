#include <cuda.h>
    
__global__
void gpuBgrToGreyscale(unsigned char* devBgr, unsigned char* devGrey, int rows, int cols)
{

    // column index
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    // row index
    const int row = blockDim.y * blockIdx.y + threadIdx.y;

    if( row >= rows || col >= cols )
    {
        return;
    }

    // Start index of grey pixel
    int idx = row * cols + col;

    // Color pixel index
    // The pixel storage format (BGR BGR BGR BGR ... ), makes this operation a
    // row-wise reduction (loosely speaking)
    int c = 3 * idx;

    devGrey[idx] = (unsigned char) (0.11402 * devBgr[c] + 0.58704 * devBgr[c + 1] + 0.29894 * devBgr[c + 2]);
}
