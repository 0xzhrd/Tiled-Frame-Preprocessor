cmake_minimum_required(VERSION 3.20)
project(tiled_frame_preprocessor LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

option(FRAME_FUSE_RN "Fuse resize+normalize+transpose for production" ON)

if(FRAME_FUSE_RN)
    set(KERNEL_SOURCES src/colour_convert_kernel.cu
                       src/normalize_transpose_kernel.cu)
else()
    set(KERNEL_SOURCES src/colour_convert_kernel.cu
                       src/resize_kernel.cu
                       src/normalize_transpose_kernel.cu)

endif()

add_library(frame_preprocessor STATIC src/preprocessor.cu src/pipelined_preprocessor.cu ${KERNEL_SOURCES})
target_include_directories(frame_preprocessor PUBLIC include)
if(FRAME_FUSE_RN)
    target_compile_definitions(frame_preprocessor PUBLIC FRAME_FUSE_RN)
endif()
set_target_properties(frame_preprocessor PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(frame_preprocessor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)

add_executable(test_correctness test/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE frame_preprocessor ${OpenCV_LIBS})
target_include_directories(test_correctness PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(benchmarks benchmarks/benchmarks.cu)
target_link_libraries(benchmarks PRIVATE frame_preprocessor)

add_executable(pipelined_benchmark benchmarks/pipelined_benchmark.cu)
target_link_libraries(pipelined_benchmark PRIVATE frame_preprocessor)
target_include_directories(pipelined_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(detect_humans apps/detect_humans.cu src/yolo_detector.cu)
target_link_libraries(detect_humans PRIVATE frame_preprocessor nvinfer cudart)
target_include_directories(detect_humans PRIVATE ${TensorRT_INCLUDE_DIR})
target_link_directories(detect_humans PRIVATE ${TensorRT_LIB_DIR})

enable_testing()
add_test(NAME correctness COMMAND test_correctness)
