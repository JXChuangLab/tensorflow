#version 310 es

// 纹理到SSBO转换的计算着色器
// 将RGBA纹理数据转换为BHWC格式的SSBO

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// 输入纹理
layout(binding = 0) uniform sampler2D input_texture;

// 输出SSBO
layout(std430, binding = 1) buffer OutputBuffer {
    float output_data[];
};

// 输入参数
uniform int input_width;
uniform int input_height;
uniform int input_channels;
uniform int output_width;
uniform int output_height;
uniform int output_channels;

void main() {
    ivec2 tex_coord = ivec2(gl_GlobalInvocationID.xy);
    
    // 检查边界
    if (tex_coord.x >= input_width || tex_coord.y >= input_height) {
        return;
    }
    
    // 读取纹理数据
    vec4 texel = texelFetch(input_texture, tex_coord, 0);
    
    // 计算输出索引
    int output_index = tex_coord.y * output_width + tex_coord.x;
    
    // 根据通道数输出数据
    if (output_channels == 1) {
        // 灰度输出
        output_data[output_index] = texel.r;
    } else if (output_channels == 3) {
        // RGB输出
        output_data[output_index * 3 + 0] = texel.r;
        output_data[output_index * 3 + 1] = texel.g;
        output_data[output_index * 3 + 2] = texel.b;
    } else if (output_channels == 4) {
        // RGBA输出
        output_data[output_index * 4 + 0] = texel.r;
        output_data[output_index * 4 + 1] = texel.g;
        output_data[output_index * 4 + 2] = texel.b;
        output_data[output_index * 4 + 3] = texel.a;
    }
}