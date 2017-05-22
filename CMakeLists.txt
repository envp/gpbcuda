cmake_minimum_required(VERSION 2.8)
project( gpbcuda )

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

add_executable( gpbcuda main.cpp )

target_link_libraries( gpbcuda 
    ${OpenCV_LIBS} 
    ${CUDA_LIBRARIES}    
)
