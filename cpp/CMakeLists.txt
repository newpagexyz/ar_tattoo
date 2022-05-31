cmake_minimum_required(VERSION 3.5)

project(opencv-ar-tattoo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 3.3 REQUIRED)
find_package(OpenCV REQUIRED )

add_executable(ar-tattoo src/main.cpp src/glad.c src/shaders/shader.cpp)
target_link_libraries(ar-tattoo ${OpenCV_LIBRARIES} glfw ${CMAKE_DL_LIBS})
