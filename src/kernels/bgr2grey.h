#ifndef GLOBALPB_SRC_KERNELS_BGR2GRAY_H
#define GLOBALPB_SRC_KERNELS_BGR2GRAY_H

__global__ void gpuBgrToGreyscale(unsigned char* devBgr, unsigned char* devGrey, int rows, int cols);

#endif /* GLOBALPB_SRC_KERNELS_BGR2GRAY_H */
