cmake_minimum_required(VERSION 3.26)
project(Geny)

set(CMAKE_CXX_STANDARD 17)

# Dodaj flagę kompilacji dla OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(Geny
        Genetic.cpp
)
