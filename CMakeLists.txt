cmake_minimum_required(VERSION 3.17)
project(voxel LANGUAGES CUDA CXX)

set(VOXEL_SRC src/voxel.cu)

set(CMAKE_CUDA_ARCHITECTURES 72) # TODO find dynamically

#if(NOT DEFINED CMAKE_CUDA_STANDARD)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif()


set(CUDA_INCLUDE_DIRS /usr/local/cuda/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include/)


set(VOXEL_INCLUDES
  ${PCL_INCLUDE_DIRS}
  include
)

set(VOXEL_LIBS
  ${PCL_LIBRARIES}
)

list(APPEND VOXEL_LIBS
  ${CUDA_LIBRARIES}
)
list(APPEND VOXEL_INCLUDES
  ${CUDA_INCLUDE_DIRS}
  ext/cuCollections/cuCollections/include
)

if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${VOXEL_SRC})
else()
  add_library(${PROJECT_NAME} STATIC ${VOXEL_SRC})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${VOXEL_INCLUDES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${VOXEL_LIBS})