// 使用示例：TFLite GPU推理包装器
#include "lite_wrapper.h"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "TFLite GPU Wrapper 版本: " << TfLiteWrapperVersion() << std::endl;

    // 1. 创建GPU模型
    const char* model_path = "path/to/your/model.tflite";
    TfLiteGpuModel model = TfLiteGpuModelCreate(model_path, TFLITE_GPU_PRIORITY_MAX_PRECISION);
    
    if (!model) {
        std::cerr << "无法创建GPU模型" << std::endl;
        return -1;
    }

    std::cout << "GPU模型创建成功" << std::endl;

    // 2. 示例1：使用纹理输入进行推理
    std::cout << "\n=== 纹理输入推理示例 ===" << std::endl;
    
    // 假设我们有一个OpenGL纹理ID
    uint32_t texture_id = 1;  // 实际使用时需要从OpenGL获取
    int width = 224;
    int height = 224;
    
    bool success = TfLiteGpuModelInvokeTexture(model, texture_id, width, height);
    if (success) {
        std::cout << "纹理推理成功" << std::endl;
        
        // 获取输出
        TfLiteOutputs outputs;
        TfLiteOutput output_buffer[10];  // 假设最多10个输出
        outputs.outputs = output_buffer;
        outputs.size = 10;
        
        if (TfLiteGpuModelGetOutput(model, &outputs)) {
            std::cout << "输出数量: " << outputs.size << std::endl;
            for (int i = 0; i < outputs.size; ++i) {
                std::cout << "输出 " << i << ": "
                          << "尺寸=" << output_buffer[i].width << "x" 
                          << output_buffer[i].height << "x" 
                          << output_buffer[i].channels
                          << ", 元素数=" << output_buffer[i].size << std::endl;
            }
        }
    } else {
        std::cerr << "纹理推理失败" << std::endl;
    }

    // 3. 示例2：使用缓冲区输入进行推理
    std::cout << "\n=== 缓冲区输入推理示例 ===" << std::endl;
    
    // 创建测试数据
    int input_width = 224;
    int input_height = 224;
    int input_channels = 3;
    int input_size = input_width * input_height * input_channels;
    
    std::vector<float> input_data(input_size);
    // 填充测试数据（这里用随机数据）
    for (int i = 0; i < input_size; ++i) {
        input_data[i] = static_cast<float>(i) / input_size;
    }
    
    success = TfLiteGpuModelInvokeBuffer(model, input_data.data(), 
                                        input_width, input_height, input_channels);
    if (success) {
        std::cout << "缓冲区推理成功" << std::endl;
        
        // 获取输出
        TfLiteOutputs outputs;
        TfLiteOutput output_buffer[10];
        outputs.outputs = output_buffer;
        outputs.size = 10;
        
        if (TfLiteGpuModelGetOutput(model, &outputs)) {
            std::cout << "输出数量: " << outputs.size << std::endl;
            for (int i = 0; i < outputs.size; ++i) {
                std::cout << "输出 " << i << ": "
                          << "尺寸=" << output_buffer[i].width << "x" 
                          << output_buffer[i].height << "x" 
                          << output_buffer[i].channels
                          << ", 元素数=" << output_buffer[i].size << std::endl;
                
                // 打印前几个输出值
                if (output_buffer[i].data && output_buffer[i].size > 0) {
                    std::cout << "  前5个值: ";
                    for (int j = 0; j < std::min(5, output_buffer[i].size); ++j) {
                        std::cout << output_buffer[i].data[j] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    } else {
        std::cerr << "缓冲区推理失败" << std::endl;
    }

    // 4. 示例3：使用SSBO进行推理（高级用法）
    std::cout << "\n=== SSBO推理示例 ===" << std::endl;
    
    // 绑定输入SSBO（假设已经创建了SSBO）
    uint32_t input_ssbo_id = 2;  // 实际使用时需要从OpenGL获取
    if (TfLiteGpuModelBindInputSSBO(model, input_ssbo_id)) {
        std::cout << "输入SSBO绑定成功" << std::endl;
    }
    
    // 绑定输出SSBO
    uint32_t output_ssbo_id = 3;  // 实际使用时需要从OpenGL获取
    if (TfLiteGpuModelBindOutputSSBO(model, 0, output_ssbo_id)) {
        std::cout << "输出SSBO绑定成功" << std::endl;
    }
    
    // 执行推理（使用绑定的SSBO）
    success = TfLiteGpuModelInvokeTexture(model, 0, 0, 0);  // 纹理ID为0表示使用SSBO
    if (success) {
        std::cout << "SSBO推理成功" << std::endl;
    } else {
        std::cerr << "SSBO推理失败" << std::endl;
    }

    // 5. 清理资源
    TfLiteGpuModelDelete(model);
    std::cout << "\n资源清理完成" << std::endl;

    return 0;
}