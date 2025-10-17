#!/bin/bash

# TFLite GPU Wrapper 构建脚本

set -e

echo "开始构建 TFLite GPU Wrapper..."

# 检查依赖
echo "检查依赖..."

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到CMake，请先安装CMake"
    exit 1
fi

# 检查OpenGL开发包
if ! pkg-config --exists opengl; then
    echo "警告: 未找到OpenGL开发包，可能影响构建"
fi

# 检查EGL开发包
if ! pkg-config --exists egl; then
    echo "警告: 未找到EGL开发包，可能影响构建"
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build
cd build

# 配置CMake
echo "配置CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-fvisibility=hidden -fvisibility-inlines-hidden"

# 编译
echo "开始编译..."
make -j$(nproc)

echo "构建完成！"
echo ""
echo "生成的文件："
echo "  - libtflite_gpu_wrapper.so (共享库)"
echo "  - example_usage (示例程序)"
echo "  - test_wrapper (测试程序)"
echo ""
echo "运行测试："
echo "  ./test_wrapper"
echo ""
echo "运行示例："
echo "  ./example_usage"