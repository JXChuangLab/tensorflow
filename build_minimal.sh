#!/bin/bash

# TFLite GPU Wrapper 最小版本构建脚本

set -e

echo "开始构建 TFLite GPU Wrapper (最小版本)..."

# 检查依赖
echo "检查依赖..."

# 检查CMake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到CMake，请先安装CMake"
    exit 1
fi

# 检查编译器
if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到g++编译器，请先安装g++"
    exit 1
fi

# 创建构建目录
echo "创建构建目录..."
mkdir -p build_minimal
cd build_minimal

# 复制CMakeLists文件和源文件
cp ../CMakeLists_minimal.txt ./CMakeLists.txt
cp ../lite_wrapper_simple.cpp .
cp ../lite_wrapper.h .
cp ../example_usage.cpp .
cp ../test_wrapper.cpp .

# 配置CMake
echo "配置CMake..."
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_CXX_FLAGS="-fvisibility=hidden -fvisibility-inlines-hidden" \
    .

# 编译
echo "开始编译..."
make -j$(nproc)

echo "构建完成！"
echo ""
echo "生成的文件："
echo "  - libtflite_gpu_wrapper_minimal.so (共享库)"
echo "  - example_usage_minimal (示例程序)"
echo "  - test_wrapper_minimal (测试程序)"
echo ""
echo "运行测试："
echo "  ./test_wrapper_minimal"
echo ""
echo "运行示例："
echo "  ./example_usage_minimal"