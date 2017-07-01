project(caffe_deploy)
cmake_minimum_required(VERSION 3.8)

if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  set(TOP_PROJECT ON)
else()
  set(TOP_PROJECT OFF)
endif()

### option and message

option(DEBUG "Debug mode." OFF)
option(CPU_ONLY "CPU only mode." OFF)
option(USE_CUDNN "Use CuDNN library." ON)
option(USE_EIGEN "Use submodules/eigen as BLAS." ON)

if(DEBUG)
  message(STATUS "Use debug mode.")
else()
  message(STATUS "Do not use debug mode.")
endif()
if(CPU_ONLY)
  message(STATUS "Use CPU only mode.")
  if(USE_CUDNN)
    message(STATUS "CuDNN is not available under CPU only mode.")
    set(USE_CUDNN OFF)
  endif()
else()
  message(STATUS "Use CPU and GPU mode.")
  if(USE_CUDNN)
    message(STATUS "Use CuDNN library.")
  else()
    message(STATUS "Do not use CuDNN library.")
  endif()
endif()
if(USE_EIGEN)
  message(STATUS "Use submodules/eigen as BLAS.")
else()
  message(STATUS "Use OpenBlas as BLAS.")
endif()

### sub directory

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/protobuf)
if(USE_EIGEN)
  set(EIGEN_CORE_FUNCTION_ONLY ON)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/submodules/eigen)
endif()

### protoc

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/proto/caffe.proto)
set(PROTO_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/proto/caffe.pb.cc)
set(PROTO_HEADER_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/caffe/proto/)
set(PROTO_HEADER_FILE caffe.pb.h)
if(CMAKE_CROSSCOMPILING)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/host_compile_protobuf)
  execute_process(COMMAND bash -c "grep processor /proc/cpuinfo | wc -l" OUTPUT_VARIABLE CPU_NUM)
  string(STRIP ${CPU_NUM} CPU_NUM)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/host_compile_protobuf/protoc
    COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/submodules/protobuf
    COMMAND make -j${CPU_NUM} protoc
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/host_compile_protobuf
  )
  set(CAFFE_PROTOC_PATH ${CMAKE_CURRENT_BINARY_DIR}/host_compile_protobuf/protoc)
else()
  set(CAFFE_PROTOC_PATH protoc)
endif()
add_custom_command(
  OUTPUT ${PROTO_SRC} ${PROTO_HEADER_PATH}/${PROTO_HEADER_FILE} DEPENDS ${CAFFE_PROTOC_PATH} ${PROTO_FILE}
  COMMAND ${CAFFE_PROTOC_PATH} ${PROTO_FILE} --cpp_out=. --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src
  COMMAND mv caffe/proto/caffe.pb.cc ${PROTO_SRC}
  COMMAND mkdir -p ${PROTO_HEADER_PATH}
  COMMAND mv caffe/proto/caffe.pb.h ${PROTO_HEADER_PATH}/${PROTO_HEADER_FILE}
)

### CUDA

if(NOT CPU_ONLY)
  if(MSVC)
    message(FATAL_ERROR "CUDA on MSVC is not supported yet.")
  endif()
  find_package(CUDA REQUIRED)
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)
  enable_language(CUDA)
endif()

### target

file(GLOB_RECURSE CPP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/*.cpp)
file(GLOB_RECURSE TEST_CPP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/test_*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/gtest/*.cpp)
list(REMOVE_ITEM CPP_SRCS ${TEST_CPP_SRCS})
list(APPEND SRCS ${CPP_SRCS} ${PROTO_SRC})
if(NOT CPU_ONLY)
  file(GLOB_RECURSE CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/*.cu)
  file(GLOB_RECURSE TEST_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/caffe/test_*.cu)
  list(REMOVE_ITEM CUDA_SRCS ${TEST_CUDA_SRCS})
  list(APPEND SRCS ${CUDA_SRCS})
endif()
if(TOP_PROJECT)
  add_library(caffe_deploy SHARED ${SRCS})
else()
  add_library(caffe_deploy STATIC ${SRCS})
endif()

### compile options and definations

target_compile_features(caffe_deploy PUBLIC cxx_std_11)
if(NOT CPU_ONLY)
  set_property(TARGET caffe_deploy PROPERTY CUDA_STANDARD_REQUIRED ON)
  set_property(TARGET caffe_deploy PROPERTY CUDA_STANDARD 11)
endif()

if(DEBUG)
  target_compile_definitions(caffe_deploy PRIVATE DEBUG)
  if(MSVC)
    target_compile_definitions(caffe_deploy PRIVATE /Od)
  else()
    target_compile_options(caffe_deploy PRIVATE -g)
    target_compile_options(caffe_deploy PRIVATE -O0)
  endif()
else()
  target_compile_definitions(caffe_deploy PRIVATE NDEBUG)
  if(MSVC)
    target_compile_options(caffe_deploy PRIVATE /O2)
  else()
    target_compile_options(caffe_deploy PRIVATE -O3)
  endif()
endif()

if(CPU_ONLY)
  target_compile_definitions(caffe_deploy PUBLIC CPU_ONLY)
else()
  if(USE_CUDNN)
    target_compile_definitions(caffe_deploy PUBLIC USE_CUDNN)
  endif()
endif()
if(USE_EIGEN)
  target_compile_definitions(caffe_deploy PUBLIC USE_EIGEN)
endif()

if(MSVC)
  target_compile_options(caffe_deploy PRIVATE -D_CRT_SECURE_NO_WARNINGS=1 /MP)
  target_compile_options(caffe_deploy PRIVATE /wd4244 /wd4267 /wd4018 /wd4355 /wd4800 /wd4251 /wd4996 /wd4146 /wd4305)
endif()

set_target_properties(caffe_deploy PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT CPU_ONLY)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_30,code=sm_30>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_32,code=sm_32>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_35,code=sm_35>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_53,code=sm_53>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_62,code=[sm_62,compute_62]>)
endif()

### include

target_include_directories(caffe_deploy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(NOT CPU_ONLY)
  target_include_directories(caffe_deploy PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  if(USE_CUDNN)
    if(DEFINED CUDNN_DIR)
      target_include_directories(caffe_deploy PUBLIC ${CUDNN_DIR}/include)
    endif()
  endif()
endif()

### link

target_link_libraries(caffe_deploy PUBLIC protobuf)
if(USE_EIGEN)
  target_link_libraries(caffe_deploy PRIVATE eigen)
endif()
find_package(Threads REQUIRED)
if(CMAKE_USE_PTHREADS_INIT)
  target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pthread>)
  if(NOT CPU_ONLY)
    target_compile_options(caffe_deploy PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>)
  endif()
  target_link_libraries(caffe_deploy PUBLIC -pthread)
endif()
if(NOT CPU_ONLY)
  target_link_libraries(caffe_deploy PUBLIC cublas)
  target_link_libraries(caffe_deploy PUBLIC curand)
  if(USE_CUDNN)
    if(DEFINED CUDNN_DIR)
      find_library(CUDNN_LIBRARY cudnn HINTS ${CUDNN_DIR} PATH_SUFFIXES lib64 NO_DEFAULT_PATH)
      if(NOT CUDNN_LIBRARY)
        message(FATAL_ERROR "No cuDNN library found in " ${CUDNN_DIR})
      else()
        message(STATUS "cuDNN library: " ${CUDNN_LIBRARY})
        target_link_libraries(caffe_deploy PUBLIC ${CUDNN_LIBRARY})
      endif()
    else()
      target_link_libraries(caffe_deploy PUBLIC cudnn)
    endif()
  endif()
endif()
if(NOT USE_EIGEN)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS library: " ${BLAS_LIBRARIES})
  target_link_libraries(caffe_deploy PUBLIC ${BLAS_LIBRARIES})
endif()

### properties

set_target_properties(caffe_deploy PROPERTIES INSTALL_INCLUDE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
set_target_properties(caffe_deploy PROPERTIES INSTALL_INCLUDE_PB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/caffe/proto/caffe.pb.h)

### install

if(TOP_PROJECT)
  set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
  install(TARGETS caffe_deploy DESTINATION lib)
  install(DIRECTORY include/caffe DESTINATION include)
  get_target_property(PROTOBUF_INSTALL_INCLUDE_PATH protobuf INSTALL_INCLUDE_PATH)
  get_target_property(PROTOBUF_INSTALL_INCLUDE_PATTERN protobuf INSTALL_INCLUDE_PATTERN)
  install(DIRECTORY ${PROTOBUF_INSTALL_INCLUDE_PATH} DESTINATION include
      FILES_MATCHING PATTERN ${PROTOBUF_INSTALL_INCLUDE_PATTERN})
endif()

### test

if(TOP_PROJECT)
  if(CPU_ONLY)
    add_executable(caffe_deploy_test ${TEST_CPP_SRCS})
    set(GTEST_FILTER "--gtest_filter=\"-*GPU*\"")
  else()
    add_executable(caffe_deploy_test ${TEST_CPP_SRCS} ${TEST_CUDA_SRCS})
    target_include_directories(caffe_deploy_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_30,code=sm_30>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_32,code=sm_32>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_35,code=sm_35>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_50,code=sm_50>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_52,code=sm_52>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_53,code=sm_53>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_60,code=sm_60>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=sm_61>)
    target_compile_options(caffe_deploy_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_62,code=[sm_62,compute_62]>)
  endif()
  target_compile_features(caffe_deploy_test PRIVATE cxx_std_11)
  target_include_directories(caffe_deploy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(caffe_deploy_test PRIVATE caffe_deploy)
  add_custom_target(runtest caffe_deploy_test 0 ${GTEST_FILTER})
endif()
