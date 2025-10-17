#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// 与现有优先级枚举保持一致
typedef enum {
  TFLITE_GPU_PRIORITY_AUTO = 0,
  TFLITE_GPU_PRIORITY_MIN_LATENCY = 1,
  TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE = 2,
  TFLITE_GPU_PRIORITY_MAX_PRECISION = 3,
} TfLiteGpuPriority;

// 句柄
typedef void* TfLiteGpuRunner;

// 创建 Runner（要求当前线程已绑定有效 EGL 上下文）
// prefer_fp16: 1=FP16(允许精度损失), 0=FP32
__attribute__((visibility("default"))) TfLiteGpuRunner
TfLiteGpuRunnerCreate(const char* model_path,
                      TfLiteGpuPriority priority,
                      int prefer_fp16);

// 使用 GL 纹理2D作为输入进行一次推理（零拷贝路径）。
// input_texture_2d: GL_TEXTURE_2D 的纹理 id；宽高需与模型输入匹配。
// 若希望将输出写入已有输出纹理，传入 output_texture_2d(可为0表示不绑定)。
__attribute__((visibility("default"))) bool
TfLiteGpuRunnerInvokeTexture(TfLiteGpuRunner runner,
                             uint32_t input_texture_2d,
                             int tex_w, int tex_h,
                             uint32_t output_texture_2d /*可为0*/);

// 读取指定输出到 CPU（如需）。注意：这会发生 GPU->CPU 拷贝。
__attribute__((visibility("default"))) bool
TfLiteGpuRunnerReadOutput(TfLiteGpuRunner runner,
                          int index, float* out, int out_count);

// 输出个数
__attribute__((visibility("default"))) int
TfLiteGpuRunnerGetNumOutputs(TfLiteGpuRunner runner);

// 销毁 Runner
__attribute__((visibility("default"))) void
TfLiteGpuRunnerDelete(TfLiteGpuRunner runner);

#ifdef __cplusplus
}  // extern "C"
#endif
