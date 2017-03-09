cmake_minimum_required(VERSION 2.6)

project(featureComparison)
set(featureComparison_VERSION_MAJOR 0)
set(featureComparison_VERSION_MINOR 1)
set(featureComparison_VERSION_PATCH 0)

set(CPACK_PACKAGE_VERSION_MAJOR "${featureComparison_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${featureComparison_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${featureComparison_VERSION_PATCH}")
set(CPACK_GENERATOR "ZIP")
include(CPack)

find_package(OpenCV REQUIRED)
find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;--compiler-options;-O2;-DVERBOSE")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -msse2 ")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;--compiler-options;-O2;-DVERBOSE")
  endif()
endif()

set(cuda_sources
  # dynamic.cu
  CudaSift/cudaImage.cu
  CudaSift/cudaImage.h
  CudaSift/cudaSiftH.cu
  CudaSift/cudaSiftH.h
  CudaSift/matching.cu
  CudaSift/cudaSiftD.h
  CudaSift/cudaSift.h
  CudaSift/cudautils.h
)

set(sources
  CudaSift/geomFuncs.cpp
  src/featureComparison.cpp
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(featurecomparison ${cuda_sources} ${sources} OPTIONS -arch=sm_35)

set_target_properties(featurecomparison PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"
)

target_link_libraries(featurecomparison
  /usr/local/cuda/lib64/libcudadevrt.a ${OpenCV_LIBS}
)

# install(FILES
#   ${cuda_sources}
#   ${sources}
#   cudaSiftD.cu
#   CMakeLists.txt
#   Copyright.txt
#   DESTINATION .
# )
# install(FILES data/left.pgm data/righ.pgm
#   DESTINATION data
# )
