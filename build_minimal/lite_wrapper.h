// -*- header -*-
#ifndef TENSORFLOW_LITE_WRAP_LITE_WRAPPER_H_
#define TENSORFLOW_LITE_WRAP_LITE_WRAPPER_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// 定义导出宏
#define TFLITE_WRAP_EXPORT __attribute__ ((visibility("default")))

// 前向声明句柄类型
typedef void* TfLiteGpuModel;

// GPU 委托优先级
typedef enum {
    TFLITE_GPU_PRIORITY_AUTO = 0,
    TFLITE_GPU_PRIORITY_MAX_PRECISION = 1,
    TFLITE_GPU_PRIORITY_MIN_LATENCY = 2,
    TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE = 3,
} TfLiteGpuPriority;

// 输入数据类型
typedef enum {
    TFLITE_INPUT_TEXTURE_2D = 0,    // GLES 纹理输入
    TFLITE_INPUT_BUFFER = 1,        // 内存缓冲区输入
} TfLiteInputType;

// 输入数据结构
typedef struct {
    TfLiteInputType type;
    union {
        uint32_t texture_id;  // 当 type = TFLITE_INPUT_TEXTURE_2D 时使用
        struct {
            const void* data; // 当 type = TFLITE_INPUT_BUFFER 时使用
            int width;
            int height;
            int channels;
        } buffer;
    };
} TfLiteInput;

// 输出数据结构
typedef struct {
    const float* data;   // 输出数据指针
    int size;           // 数据大小（元素个数）
    int width;          // 输出宽度
    int height;         // 输出高度
    int channels;       // 输出通道数
} TfLiteOutput;


typedef struct {
    TfLiteOutput* outputs; // 输出数组指针（调用方提供，size 作为容量传入）
    int size;       // 输入：容量；输出：实际填充个数
} TfLiteOutputs;

/**
 * 创建 GPU 模型实例
 * @param model_path 模型文件路径
 * @param priority GPU 优先级
 * @return 模型句柄，失败返回 NULL
 */
TFLITE_WRAP_EXPORT TfLiteGpuModel TfLiteGpuModelCreate(const char* model_path, TfLiteGpuPriority priority);

/**
 * 使用纹理进行推理
 * @param model 模型句柄
 * @param texture_id 输入纹理 ID
 * @param width 纹理宽度
 * @param height 纹理高度
 * @return 成功返回 true，失败返回 false
 */
TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeTexture(TfLiteGpuModel model, uint32_t texture_id, int width, int height);

// 如果需要走 SSBO（推荐）：在创建后绑定输入/输出 SSBO，然后调用 Invoke（用 GetOutput 从 CPU 读也可）
TFLITE_WRAP_EXPORT bool TfLiteGpuModelBindInputSSBO(TfLiteGpuModel model, uint32_t ssbo_id);
TFLITE_WRAP_EXPORT bool TfLiteGpuModelBindOutputSSBO(TfLiteGpuModel model, int index, uint32_t ssbo_id);

/**
 * 使用缓冲区进行推理（CPU 内存输入，期望 float32 BHWC）
 */
TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeBuffer(TfLiteGpuModel model, const void* input_data, int width, int height, int channels);

/**
 * 获取推理输出（CPU 指针，不拷贝数据；调用方提供 outputs->outputs 缓冲及容量）
 */
TFLITE_WRAP_EXPORT bool TfLiteGpuModelGetOutput(TfLiteGpuModel model, TfLiteOutputs* outputs);

/** 释放模型资源 */
TFLITE_WRAP_EXPORT void TfLiteGpuModelDelete(TfLiteGpuModel model);

/** 获取库版本号 */
TFLITE_WRAP_EXPORT const char* TfLiteWrapperVersion();

#ifdef __cplusplus
}
#endif

#endif  // TENSORFLOW_LITE_WRAP_LITE_WRAPPER_H_