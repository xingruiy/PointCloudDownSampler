cmake_minimum_required(VERSION 3.0)
project(PCDSampler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
find_package(InputFactory REQUIRED)

add_library(${PROJECT_NAME} SHARED "")

target_sources(${PROJECT_NAME}
PRIVATE
    PointCloudSampler.cu
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${Pangolin_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${OpenCV_LIBS}
    InputFactory::InputFactory
    ${Pangolin_LIBRARIES}
)

add_executable(sampler main.cpp)
target_link_libraries(sampler PUBLIC ${PROJECT_NAME})