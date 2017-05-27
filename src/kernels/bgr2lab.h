#ifndef GLOBALPB_SRC_KERNELS_BGR2LAB_H
#define GLOBALPB_SRC_KERNELS_BGR2LAB_H

__global__ void gpuBgrToLab(unsigned char* devBgr, unsigned char* devL, unsigned char* devA, unsigned char* devB, int rows, int cols, float gamma);

#endif /* GLOBALPB_SRC_KERNELS_BGR2LAB_H */
