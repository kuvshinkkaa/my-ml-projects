cmake_minimum_required(VERSION 3.10)
project(TwoStageDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio objdetect dnn)

add_executable(detector main.cpp)
target_include_directories(detector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(detector PRIVATE ${OpenCV_LIBS})
