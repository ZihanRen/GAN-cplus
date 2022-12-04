cmake_minimum_required(VERSION 3.2)

project(GAN-cplus)
add_definitions(-std=c++11)

# Enable C++11
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# set(CMAKE_PREFIX_PATH /home/topaz/repo/libtorch)
list(APPEND CMAKE_PREFIX_PATH "/home/topaz/repo/libtocrch")
# list(APPEND CMAKE_PREFIX_PATH "/usr/local/share/opencv4")
list(APPEND CMAKE_PREFIX_PATH "/usr/local/lib/cmake/opencv4")

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED PATHS /usr/local/include/opencv4)
find_package(OpenCV REQUIRED)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
