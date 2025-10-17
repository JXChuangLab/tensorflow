# TFLite GPU 推理包装器

基于MediaPipe的TFLite GPU推理包装器，支持纹理输入和SSBO（Shader Storage Buffer Object）进行GPU推理。

## 功能特性

- ✅ 支持OpenGL纹理输入
- ✅ 支持CPU内存缓冲区输入
- ✅ 支持SSBO绑定进行GPU推理
- ✅ 基于MediaPipe的TFLite GPU Runner
- ✅ 支持多种GPU优先级设置
- ✅ 完整的C接口，易于集成

## 依赖项

- TensorFlow Lite
- MediaPipe
- OpenGL ES 3.1+ 或 OpenGL 4.3+
- EGL
- Abseil库

## 构建

```bash
mkdir build
cd build
cmake ..
make -j4
```

## 使用方法

### 基本使用

```cpp
#include "lite_wrapper.h"

// 1. 创建GPU模型
TfLiteGpuModel model = TfLiteGpuModelCreate("model.tflite", TFLITE_GPU_PRIORITY_MAX_PRECISION);

// 2. 使用纹理进行推理
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

### 纹理输入推理

```cpp
// 假设你有一个OpenGL纹理
uint32_t texture_id = get_opengl_texture_id();
int width = 224;
int height = 224;

// 执行推理
bool success = TfLiteGpuModelInvokeTexture(model, texture_id, width, height);
```

### 缓冲区输入推理

```cpp
// 准备输入数据（BHWC格式）
std::vector<float> input_data(width * height * channels);
// ... 填充数据 ...

// 执行推理
bool success = TfLiteGpuModelInvokeBuffer(model, input_data.data(), width, height, channels);
```

### SSBO绑定推理（高级用法）

```cpp
// 绑定输入SSBO
TfLiteGpuModelBindInputSSBO(model, input_ssbo_id);

// 绑定输出SSBO
TfLiteGpuModelBindOutputSSBO(model, 0, output_ssbo_id);

// 执行推理（使用绑定的SSBO）
TfLiteGpuModelInvokeTexture(model, 0, 0, 0);  // 纹理ID为0表示使用SSBO
```

## API 参考

### 数据类型

#### TfLiteGpuPriority
GPU推理优先级：
- `TFLITE_GPU_PRIORITY_AUTO`: 自动选择
- `TFLITE_GPU_PRIORITY_MAX_PRECISION`: 最大精度
- `TFLITE_GPU_PRIORITY_MIN_LATENCY`: 最小延迟
- `TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE`: 最小内存使用

#### TfLiteInputType
输入数据类型：
- `TFLITE_INPUT_TEXTURE_2D`: OpenGL纹理输入
- `TFLITE_INPUT_BUFFER`: 内存缓冲区输入

#### TfLiteInput
输入数据结构，支持纹理和缓冲区两种输入方式。

#### TfLiteOutput
输出数据结构，包含输出数据指针和维度信息。

### 函数接口

#### TfLiteGpuModelCreate
```cpp
TfLiteGpuModel TfLiteGpuModelCreate(const char* model_path, TfLiteGpuPriority priority);
```
创建GPU模型实例。

**参数：**
- `model_path`: TFLite模型文件路径
- `priority`: GPU推理优先级

**返回值：**
- 成功：模型句柄
- 失败：NULL

#### TfLiteGpuModelInvokeTexture
```cpp
bool TfLiteGpuModelInvokeTexture(TfLiteGpuModel model, uint32_t texture_id, int width, int height);
```
使用OpenGL纹理进行推理。

**参数：**
- `model`: 模型句柄
- `texture_id`: OpenGL纹理ID
- `width`: 纹理宽度
- `height`: 纹理高度

**返回值：**
- 成功：true
- 失败：false

#### TfLiteGpuModelInvokeBuffer
```cpp
bool TfLiteGpuModelInvokeBuffer(TfLiteGpuModel model, const void* input_data, int width, int height, int channels);
```
使用内存缓冲区进行推理。

**参数：**
- `model`: 模型句柄
- `input_data`: 输入数据指针（float32 BHWC格式）
- `width`: 输入宽度
- `height`: 输入高度
- `channels`: 输入通道数

**返回值：**
- 成功：true
- 失败：false

#### TfLiteGpuModelBindInputSSBO
```cpp
bool TfLiteGpuModelBindInputSSBO(TfLiteGpuModel model, uint32_t ssbo_id);
```
绑定输入SSBO。

#### TfLiteGpuModelBindOutputSSBO
```cpp
bool TfLiteGpuModelBindOutputSSBO(TfLiteGpuModel model, int index, uint32_t ssbo_id);
```
绑定输出SSBO。

#### TfLiteGpuModelGetOutput
```cpp
bool TfLiteGpuModelGetOutput(TfLiteGpuModel model, TfLiteOutputs* outputs);
```
获取推理输出。

**参数：**
- `model`: 模型句柄
- `outputs`: 输出结构体指针

**返回值：**
- 成功：true
- 失败：false

#### TfLiteGpuModelDelete
```cpp
void TfLiteGpuModelDelete(TfLiteGpuModel model);
```
释放模型资源。

#### TfLiteWrapperVersion
```cpp
const char* TfLiteWrapperVersion();
```
获取库版本号。

## 实现细节

### 纹理到SSBO转换

包装器使用计算着色器将OpenGL纹理数据转换为SSBO格式，支持以下转换：

- RGBA纹理 → BHWC格式SSBO
- 支持1、3、4通道输出
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

## 示例程序

项目包含以下示例程序：

- `example_usage.cpp`: 基本使用示例
- `test_wrapper.cpp`: 单元测试程序

## 注意事项

1. 确保OpenGL上下文已正确初始化
2. 纹理格式应为RGBA32F或兼容格式
3. 输入数据应为BHWC格式（Batch, Height, Width, Channels）
4. 输出数据通过SSBO映射，注意内存访问权限

## 许可证

Apache License 2.0