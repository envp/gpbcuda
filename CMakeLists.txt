cmake_minimum_required(VERSION 2.8)
project( gpbcuda )

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -O3 -gencode arch=compute_35,code=sm_35
)

cuda_add_executable(
    gpbcuda
    globalPb.cu
    globalPb.h
)

target_link_libraries( gpbcuda
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)
