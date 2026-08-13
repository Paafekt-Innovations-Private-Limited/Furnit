cmake_minimum_required(VERSION 3.22.1)

project(furnit_rtmdet LANGUAGES CXX)

add_library(furnit_rtmdet SHARED rtmdet_mask_head.cpp)

# -ffast-math is deliberately NOT enabled: the mask head must stay numerically
# comparable to the Kotlin reference implementation that validates it.
target_compile_options(furnit_rtmdet PRIVATE
        -O3
        -fvisibility=hidden
        -Wall
        -Wextra
)

target_link_libraries(furnit_rtmdet log)
