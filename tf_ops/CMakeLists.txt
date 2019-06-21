cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(tf_ops LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Custom finders path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

# This exports the following variables:
# TensorFlow_INCLUDE_DIR
# TensorFlow_DIR
# TensorFlow_CXX_ABI
# TensorFlow_GIT_VERSION
# TensorFlow_VERSION
find_package(TensorFlow REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${TensorFlow_CXX_ABI}")

link_directories(
        ${TensorFlow_DIR}
        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
)
include_directories(
        ${TensorFlow_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# approxmatch
add_library(tf_approxmatch SHARED approxmatch/tf_approxmatch_g.cu approxmatch/tf_approxmatch.cpp)
target_compile_features(tf_approxmatch PUBLIC cxx_std_11)
target_link_libraries(tf_approxmatch cudart tensorflow_framework)
set_target_properties(tf_approxmatch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# nn_distance
add_library(tf_nndistance SHARED nn_distance/tf_nndistance_g.cu nn_distance/tf_nndistance.cpp)
target_compile_features(tf_nndistance PUBLIC cxx_std_11)
target_link_libraries(tf_nndistance cudart tensorflow_framework)
set_target_properties(tf_nndistance PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
