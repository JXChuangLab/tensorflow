#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef TFLITE_WRAP_EXPORT
#define TFLITE_WRAP_EXPORT __attribute__((visibility("default")))
#endif

// GPU 优先级（与 TFLite GPU 语义对齐）
typedef enum {
  TFLITE_GPU_PRIORITY_AUTO = 0,
  TFLITE_GPU_PRIORITY_MIN_LATENCY = 1,
  TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE = 2,
  TFLITE_GPU_PRIORITY_MAX_PRECISION = 3,
} TfLiteGpuPriority;

// 输出结构（CPU 读取用；GPU 零拷贝场景通常不需要）
typedef struct TfLiteOutput {
  float* data;     // 输出数据指针（float）
  int size;        // 元素个数（float 计）
  int width;
  int height;
  int channels;
} TfLiteOutput;

#ifndef TFLITE_MAX_OUTPUTS
#define TFLITE_MAX_OUTPUTS 8
#endif

typedef struct TfLiteOutputs {
  int size;  // 实际输出数量
  TfLiteOutput outputs[TFLITE_MAX_OUTPUTS];
} TfLiteOutputs;

// 句柄类型
typedef void* TfLiteGpuModel;

// 创建基于 GPU(InferenceRunner) 的模型运行器
// prefer_fp16 缺省走 FP16（允许精度损失）
TFLITE_WRAP_EXPORT TfLiteGpuModel TfLiteGpuModelCreate(
    const char* model_path, TfLiteGpuPriority priority);

// 以 GL_TEXTURE_2D 纹理作为输入进行一次推理（零拷贝）。
// width/height 必须与模型输入一致；output_texture_2d 可传 0 表示不绑定输出。
TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeTexture(
    TfLiteGpuModel model, uint32_t texture_id, int width, int height);

// 使用 CPU 缓冲作为输入进行一次推理（如需）。GPU 路径不保证零拷贝。
TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeBuffer(
    TfLiteGpuModel model, const void* input_data, int width, int height, int channels);

// 读取所有输出到 CPU（如需）。GPU 零拷贝场景一般不调用。返回 false 表示未实现或失败。
TFLITE_WRAP_EXPORT bool TfLiteGpuModelGetOutputs(
    TfLiteGpuModel model, TfLiteOutputs* outputs);

// 释放
TFLITE_WRAP_EXPORT void TfLiteGpuModelDelete(TfLiteGpuModel model);

// 版本
TFLITE_WRAP_EXPORT const char* TfLiteWrapperVersion();

#ifdef __cplusplus
}  // extern "C"
#endif
