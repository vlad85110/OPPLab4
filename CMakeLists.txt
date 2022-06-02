cmake_minimum_required(VERSION 3.21)
project(OPPLab4 C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_COMPILER /opt/local/bin/mpicc-mpich-mp)

add_executable(OPPLab4 main.c iterations.c)
