set(CMAKE_CXX_STANDAND 17)

find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" )


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

## tensorrt 和 cuda 要修改
# tensorrt
include_directories("/home/yiyu/NVIDIA/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/include")
link_directories("/home/yiyu/NVIDIA/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib")


find_package(CUDA REQUIRED)
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

aux_source_directory(kalman_filter kalman_src)
aux_source_directory(detect_armour detect_src)

message("${kalman_src}")
add_library(modules SHARED ${kalman_src} ${detect_src} )

target_link_libraries(modules
    ${OpenCV_LIBS}
    fmt::fmt nvinfer cudart
)
