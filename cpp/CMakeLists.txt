cmake_minimum_required(VERSION 3.15)
project(ACT_OPENVINO_CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO QUIET)

if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found: ${OpenVINO_VERSION}")

    set(SOURCES
        src/act_inference.cpp
        src/main.cpp
    )

    add_executable(act_inference ${SOURCES})

    target_include_directories(act_inference PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
    )

    target_link_libraries(act_inference
        openvino::runtime
        ${OpenCV_LIBS}
    )

    install(TARGETS act_inference DESTINATION bin)

    add_executable(benchmark_cpu_gpu src/benchmark_cpu_gpu.cpp)
    target_link_libraries(benchmark_cpu_gpu
        openvino::runtime
        ${OpenCV_LIBS}
    )

    install(TARGETS benchmark_cpu_gpu DESTINATION bin)
else()
    message(STATUS "OpenVINO not found. Cannot build executables.")
endif()
