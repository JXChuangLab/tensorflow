cmake_minimum_required(VERSION 3.16)
project(TfLiteGpuWrapperSimple)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

# 创建共享库
add_library(tflite_gpu_wrapper_simple SHARED
    lite_wrapper_simple.cpp
)

# 设置库的导出符号
set_target_properties(tflite_gpu_wrapper_simple PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    PUBLIC_HEADER lite_wrapper.h
)

# 创建示例程序
add_executable(example_usage_simple example_usage.cpp)
target_link_libraries(example_usage_simple tflite_gpu_wrapper_simple)

# 创建测试程序
add_executable(test_wrapper_simple test_wrapper.cpp)
target_link_libraries(test_wrapper_simple tflite_gpu_wrapper_simple)

# 安装规则
install(TARGETS tflite_gpu_wrapper_simple
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

# 如果支持，添加测试
enable_testing()
add_test(NAME test_wrapper_simple COMMAND test_wrapper_simple)