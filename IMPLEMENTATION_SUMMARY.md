# TFLite GPU 推理包装器实现总结

## 项目概述

基于MediaPipe的TFLite GPU推理代码，实现了一个支持纹理输入的GPU推理包装器。该包装器提供了简洁的C接口，支持OpenGL纹理输入和SSBO（Shader Storage Buffer Object）进行GPU推理。

## 实现文件

### 核心文件

1. **`lite_wrapper.h`** - C接口头文件
   - 定义了所有公共API接口
   - 包含数据结构定义（TfLiteInput, TfLiteOutput等）
   - 支持纹理输入和缓冲区输入

2. **`lite_wrapper.cpp`** - 完整实现版本
   - 基于MediaPipe的TFLite GPU Runner
   - 支持OpenGL纹理到SSBO转换
   - 包含计算着色器实现
   - 需要完整的MediaPipe和TensorFlow Lite依赖

3. **`lite_wrapper_simple.cpp`** - 简化实现版本
   - 不依赖MediaPipe的复杂依赖
   - 提供框架和接口演示
   - 适合快速原型开发

### 构建文件

4. **`CMakeLists.txt`** - 完整版本构建配置
   - 包含所有依赖项（OpenGL, EGL, TensorFlow Lite, MediaPipe）
   - 适合生产环境使用

5. **`CMakeLists_minimal.txt`** - 最小版本构建配置
   - 不依赖外部库
   - 适合测试和演示

### 示例和测试

6. **`example_usage.cpp`** - 使用示例
   - 演示纹理输入推理
   - 演示缓冲区输入推理
   - 演示SSBO绑定推理

7. **`test_wrapper.cpp`** - 单元测试
   - 测试所有API接口
   - 测试错误处理
   - 测试内存管理

8. **`texture_to_ssbo_shader.glsl`** - 计算着色器
   - 将OpenGL纹理转换为SSBO格式
   - 支持多种通道数（1, 3, 4通道）

### 构建脚本

9. **`build.sh`** - 完整版本构建脚本
10. **`build_minimal.sh`** - 最小版本构建脚本

## 核心功能

### 1. 模型管理
- `TfLiteGpuModelCreate()` - 创建GPU模型实例
- `TfLiteGpuModelDelete()` - 释放模型资源

### 2. 推理接口
- `TfLiteGpuModelInvokeTexture()` - 纹理输入推理
- `TfLiteGpuModelInvokeBuffer()` - 缓冲区输入推理

### 3. SSBO绑定
- `TfLiteGpuModelBindInputSSBO()` - 绑定输入SSBO
- `TfLiteGpuModelBindOutputSSBO()` - 绑定输出SSBO

### 4. 输出获取
- `TfLiteGpuModelGetOutput()` - 获取推理输出

## 技术特点

### 纹理到SSBO转换
- 使用计算着色器将OpenGL纹理数据转换为BHWC格式的SSBO
- 支持RGBA纹理到不同通道数的转换
- 自动处理数据格式转换

### GPU推理流程
1. 初始化TFLite GPU Runner
2. 创建输入/输出SSBO
3. 将纹理数据转换为SSBO格式
4. 绑定SSBO到TFLite GPU Runner
5. 执行GPU推理
6. 从输出SSBO读取结果

### 内存管理
- 所有GPU资源在GL上下文中管理
- 自动清理SSBO和着色器资源
- 支持多线程安全（每个线程独立的GL上下文）

## 构建和测试

### 最小版本构建（推荐用于测试）
```bash
cd /workspace
./build_minimal.sh
cd build_minimal
./test_wrapper_minimal    # 运行测试
./example_usage_minimal   # 运行示例
```

### 完整版本构建（需要MediaPipe依赖）
```bash
cd /workspace
./build.sh
```

## 使用示例

```cpp
#include "lite_wrapper.h"

// 1. 创建GPU模型
TfLiteGpuModel model = TfLiteGpuModelCreate("model.tflite", TFLITE_GPU_PRIORITY_MAX_PRECISION);

// 2. 纹理推理
bool success = TfLiteGpuModelInvokeTexture(model, texture_id, 224, 224);

// 3. 获取输出
TfLiteOutputs outputs;
TfLiteOutput output_buffer[10];
outputs.outputs = output_buffer;
outputs.size = 10;
TfLiteGpuModelGetOutput(model, &outputs);

// 4. 清理资源
TfLiteGpuModelDelete(model);
```

## 依赖项

### 最小版本
- C++17编译器
- CMake 3.16+

### 完整版本
- TensorFlow Lite
- MediaPipe
- OpenGL ES 3.1+ 或 OpenGL 4.3+
- EGL
- Abseil库

## 注意事项

1. 确保OpenGL上下文已正确初始化
2. 纹理格式应为RGBA32F或兼容格式
3. 输入数据应为BHWC格式（Batch, Height, Width, Channels）
4. 输出数据通过SSBO映射，注意内存访问权限

## 文件结构

```
/workspace/
├── lite_wrapper.h                    # C接口头文件
├── lite_wrapper.cpp                  # 完整实现
├── lite_wrapper_simple.cpp           # 简化实现
├── CMakeLists.txt                    # 完整版本构建配置
├── CMakeLists_minimal.txt            # 最小版本构建配置
├── example_usage.cpp                 # 使用示例
├── test_wrapper.cpp                  # 单元测试
├── texture_to_ssbo_shader.glsl       # 计算着色器
├── build.sh                          # 完整版本构建脚本
├── build_minimal.sh                  # 最小版本构建脚本
├── README.md                         # 详细文档
└── IMPLEMENTATION_SUMMARY.md         # 实现总结
```

## 下一步工作

1. 集成真实的MediaPipe依赖
2. 实现完整的纹理到SSBO转换
3. 添加更多测试用例
4. 优化性能
5. 添加错误处理和日志记录
6. 支持更多输入格式

## 许可证

Apache License 2.0