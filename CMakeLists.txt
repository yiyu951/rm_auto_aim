cmake_minimum_required(VERSION 3.15)

project(ma)
set(CMAKE_CXX_STANDAND 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g  -pthread -Wl,--no-as-needed" )


find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)


find_package(CUDA REQUIRED)

## tensorrt 和 cuda 要修改
# tensorrt
include_directories("/home/yiyu/NVIDIA/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/include")
link_directories("/home/yiyu/NVIDIA/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1/lib")

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


message("init----------------------------")
message("Eigen: Version=${Eigen3_VERSION}, ${EIGEN3_INCLUDE_DIRS}")
message("OpenCV: Version=${OpenCV_VERSION}, ${OpenCV_INCLUDE_DIRS}")
message("${PROJECT_SOURCE_DIR}")
message("end-----------------------------")


include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# PROJECT_DIR这个量，就可以在代码中使用
add_compile_definitions(PROJECT_DIR="${PROJECT_SOURCE_DIR}")

add_executable(main main.cpp)

# target_link_libraries(main
#     ${OpenCV_LIBS}
#     fmt::fmt
# )

include_directories(utils)
add_subdirectory(utils)
# target_link_libraries(main utils)

include_directories(modules)
add_subdirectory(modules)
# target_link_libraries(main modules)

include_directories(devices)
add_subdirectory(devices)
# target_link_libraries(main devices)




# add_subdirectory(utility)


target_link_libraries(main PRIVATE 
utils
modules
devices
${OpenCV_LIBS}
fmt::fmt

# utility 
)

