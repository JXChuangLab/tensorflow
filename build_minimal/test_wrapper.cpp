// 测试程序：TFLite GPU推理包装器
#include "lite_wrapper.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>

// 测试辅助函数
bool TestModelCreation() {
    std::cout << "测试模型创建..." << std::endl;
    
    // 测试无效路径
    TfLiteGpuModel model = TfLiteGpuModelCreate("nonexistent_model.tflite", TFLITE_GPU_PRIORITY_AUTO);
    if (model != nullptr) {
        std::cerr << "错误：应该返回NULL" << std::endl;
        return false;
    }
    
    // 测试NULL路径
    model = TfLiteGpuModelCreate(nullptr, TFLITE_GPU_PRIORITY_AUTO);
    if (model != nullptr) {
        std::cerr << "错误：应该返回NULL" << std::endl;
        return false;
    }
    
    std::cout << "模型创建测试通过" << std::endl;
    return true;
}

bool TestVersion() {
    std::cout << "测试版本信息..." << std::endl;
    
    const char* version = TfLiteWrapperVersion();
    if (version == nullptr) {
        std::cerr << "错误：版本信息为NULL" << std::endl;
        return false;
    }
    
    std::cout << "版本: " << version << std::endl;
    return true;
}

bool TestInvalidOperations() {
    std::cout << "测试无效操作..." << std::endl;
    
    // 测试NULL模型的操作
    bool result = TfLiteGpuModelInvokeTexture(nullptr, 1, 224, 224);
    if (result) {
        std::cerr << "错误：NULL模型应该返回false" << std::endl;
        return false;
    }
    
    result = TfLiteGpuModelInvokeBuffer(nullptr, nullptr, 224, 224, 3);
    if (result) {
        std::cerr << "错误：NULL模型应该返回false" << std::endl;
        return false;
    }
    
    result = TfLiteGpuModelBindInputSSBO(nullptr, 1);
    if (result) {
        std::cerr << "错误：NULL模型应该返回false" << std::endl;
        return false;
    }
    
    result = TfLiteGpuModelBindOutputSSBO(nullptr, 0, 1);
    if (result) {
        std::cerr << "错误：NULL模型应该返回false" << std::endl;
        return false;
    }
    
    TfLiteOutputs outputs;
    result = TfLiteGpuModelGetOutput(nullptr, &outputs);
    if (result) {
        std::cerr << "错误：NULL模型应该返回false" << std::endl;
        return false;
    }
    
    // 测试NULL输出参数
    TfLiteGpuModel model = TfLiteGpuModelCreate("test.tflite", TFLITE_GPU_PRIORITY_AUTO);
    if (model) {
        result = TfLiteGpuModelGetOutput(model, nullptr);
        if (result) {
            std::cerr << "错误：NULL输出参数应该返回false" << std::endl;
            TfLiteGpuModelDelete(model);
            return false;
        }
        TfLiteGpuModelDelete(model);
    }
    
    std::cout << "无效操作测试通过" << std::endl;
    return true;
}

bool TestMemoryManagement() {
    std::cout << "测试内存管理..." << std::endl;
    
    // 测试删除NULL模型
    TfLiteGpuModelDelete(nullptr);  // 应该不会崩溃
    
    // 测试重复删除
    TfLiteGpuModel model = TfLiteGpuModelCreate("test.tflite", TFLITE_GPU_PRIORITY_AUTO);
    if (model) {
        TfLiteGpuModelDelete(model);
        // 注意：重复删除同一个指针是未定义行为，这里不测试
    }
    
    std::cout << "内存管理测试通过" << std::endl;
    return true;
}

bool TestDataStructures() {
    std::cout << "测试数据结构..." << std::endl;
    
    // 测试枚举值
    assert(TFLITE_GPU_PRIORITY_AUTO == 0);
    assert(TFLITE_GPU_PRIORITY_MAX_PRECISION == 1);
    assert(TFLITE_GPU_PRIORITY_MIN_LATENCY == 2);
    assert(TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE == 3);
    
    assert(TFLITE_INPUT_TEXTURE_2D == 0);
    assert(TFLITE_INPUT_BUFFER == 1);
    
    // 测试结构体大小
    assert(sizeof(TfLiteInput) > 0);
    assert(sizeof(TfLiteOutput) > 0);
    assert(sizeof(TfLiteOutputs) > 0);
    
    std::cout << "数据结构测试通过" << std::endl;
    return true;
}

bool TestInputDataStructures() {
    std::cout << "测试输入数据结构..." << std::endl;
    
    // 测试纹理输入
    TfLiteInput texture_input;
    texture_input.type = TFLITE_INPUT_TEXTURE_2D;
    texture_input.texture_id = 123;
    
    assert(texture_input.type == TFLITE_INPUT_TEXTURE_2D);
    assert(texture_input.texture_id == 123);
    
    // 测试缓冲区输入
    TfLiteInput buffer_input;
    buffer_input.type = TFLITE_INPUT_BUFFER;
    buffer_input.buffer.data = nullptr;
    buffer_input.buffer.width = 224;
    buffer_input.buffer.height = 224;
    buffer_input.buffer.channels = 3;
    
    assert(buffer_input.type == TFLITE_INPUT_BUFFER);
    assert(buffer_input.buffer.width == 224);
    assert(buffer_input.buffer.height == 224);
    assert(buffer_input.buffer.channels == 3);
    
    std::cout << "输入数据结构测试通过" << std::endl;
    return true;
}

bool TestOutputDataStructures() {
    std::cout << "测试输出数据结构..." << std::endl;
    
    // 创建测试数据
    std::vector<float> test_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    TfLiteOutput output;
    output.data = test_data.data();
    output.size = test_data.size();
    output.width = 2;
    output.height = 2;
    output.channels = 1;
    
    assert(output.data == test_data.data());
    assert(output.size == test_data.size());
    assert(output.width == 2);
    assert(output.height == 2);
    assert(output.channels == 1);
    
    // 测试输出数组
    TfLiteOutputs outputs;
    TfLiteOutput output_array[3];
    outputs.outputs = output_array;
    outputs.size = 3;
    
    assert(outputs.outputs == output_array);
    assert(outputs.size == 3);
    
    std::cout << "输出数据结构测试通过" << std::endl;
    return true;
}

int main() {
    std::cout << "开始TFLite GPU Wrapper测试..." << std::endl;
    std::cout << "版本: " << TfLiteWrapperVersion() << std::endl;
    std::cout << "========================================" << std::endl;
    
    bool all_tests_passed = true;
    
    // 运行所有测试
    all_tests_passed &= TestVersion();
    all_tests_passed &= TestDataStructures();
    all_tests_passed &= TestInputDataStructures();
    all_tests_passed &= TestOutputDataStructures();
    all_tests_passed &= TestModelCreation();
    all_tests_passed &= TestInvalidOperations();
    all_tests_passed &= TestMemoryManagement();
    
    std::cout << "========================================" << std::endl;
    if (all_tests_passed) {
        std::cout << "所有测试通过！" << std::endl;
        return 0;
    } else {
        std::cout << "部分测试失败！" << std::endl;
        return 1;
    }
}