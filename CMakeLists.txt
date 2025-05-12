cmake_minimum_required(VERSION 3.30)
project(CodeForTest2WithC)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_TOOLCHAIN_FILE "C:/Users/user/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

include_directories("C:/Users/user/vcpkg/installed/x64-windows/include")

link_directories("C:/Users/user/vcpkg/installed/x64-windows/lib")

add_executable(CodeForTest2WithC main.cpp
        gen_data.h
        gen_data.cpp
        trainer.h
        trainer.cpp)

target_link_libraries(CodeForTest2WithC PRIVATE fann)

