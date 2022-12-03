cmake_minimum_required(VERSION 3.2)

project(GAN-cplus)

set(CMAKE_PREFIX_PATH /home/topaz/repo/libtorch)
# list(APPEND CMAKE_PREFIX_PATH "/home/topaz/repo/libtocrch")
# list(APPEND CMAKE_PREFIX_PATH "/usr/local/share/opencv4")

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED PATHS /usr/local/include/opencv4)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
