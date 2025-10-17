#include "tensorflow/lite/wrap/lite_wrapper.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#define GL_NO_PROTOTYPES
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"
#undef GL_NO_PROTOTYPES
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

#if defined(__ANDROID__)
#include <android/log.h>
#define LOG_TAG "LiteWrapper"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(...)
#define LOGI(...)
#define LOGW(...)
#define LOGE(...)
#endif

namespace tg = tflite::gpu;
namespace gl = tflite::gpu::gl;

// 运行时实现
class LiteGpuModelImpl {
 public:
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<gl::InferenceEnvironment> env;
  std::unique_ptr<tflite::gpu::InferenceBuilder> builder;
  std::unique_ptr<tflite::gpu::InferenceRunner> runner;

  int input_w = 0;
  int input_h = 0;
  int input_c = 4;  // 通常对齐为 4
  bool use_fp16 = true;

  // 纹理->SSBO 转换资源
  GLuint convert_program = 0;  // compute program
  GLuint input_ssbo = 0;       // 输入 SSBO（与模型输入尺寸匹配）
  GLsizei ssbo_size_bytes = 0;
};

static void FillOptionsFromPriority(TfLiteGpuPriority pri, bool* use_fp16,
                                    gl::InferenceOptions* gl_opts) {
  gl_opts->usage = tg::InferenceUsage::SUSTAINED_SPEED;
  switch (pri) {
    case TFLITE_GPU_PRIORITY_MAX_PRECISION:
      *use_fp16 = false;
      gl_opts->priority1 = tg::InferencePriority::MAX_PRECISION;
      gl_opts->priority2 = tg::InferencePriority::MIN_LATENCY;
      gl_opts->priority3 = tg::InferencePriority::MIN_MEMORY_USAGE;
      break;
    case TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE:
      *use_fp16 = true;
      gl_opts->priority1 = tg::InferencePriority::MIN_MEMORY_USAGE;
      gl_opts->priority2 = tg::InferencePriority::MAX_PRECISION;
      gl_opts->priority3 = tg::InferencePriority::MIN_LATENCY;
      break;
    case TFLITE_GPU_PRIORITY_MIN_LATENCY:
    case TFLITE_GPU_PRIORITY_AUTO:
    default:
      *use_fp16 = true;
      gl_opts->priority1 = tg::InferencePriority::MIN_LATENCY;
      gl_opts->priority2 = tg::InferencePriority::MIN_MEMORY_USAGE;
      gl_opts->priority3 = tg::InferencePriority::MAX_PRECISION;
      break;
  }
}

extern "C" {

TFLITE_WRAP_EXPORT TfLiteGpuModel TfLiteGpuModelCreate(const char* model_path,
                                                       TfLiteGpuPriority priority) {
  if (!model_path) return nullptr;
  LOGI("Create model: %s", model_path ? model_path : "(null)");

  auto* impl = new LiteGpuModelImpl();
  impl->model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!impl->model) { LOGE("Load model failed"); delete impl; return nullptr; }

  // 将 FlatBuffer 模型转换为 GPU Graph
  tg::GraphFloat32 graph;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  auto mb = tg::BuildFromFlatBuffer(*impl->model, resolver, &graph,
                                    /*allow_quant_ops=*/true,
                                    /*apply_model_transformations=*/true);
  if (!mb.ok()) { LOGE("BuildFromFlatBuffer failed: %s", mb.message().data()); delete impl; return nullptr; }

  // 创建 GL 环境
  std::unique_ptr<gl::InferenceEnvironment> env;
  gl::InferenceEnvironmentOptions env_opts;
  gl::InferenceEnvironmentProperties props;
  auto env_status = gl::NewInferenceEnvironment(env_opts, &env, &props);
  if (!env_status.ok() || !env) { LOGE("NewInferenceEnvironment failed: %s", env_status.message().data()); delete impl; return nullptr; }

  // 选项
  gl::InferenceOptions gl_opts;
  FillOptionsFromPriority(priority, &impl->use_fp16, &gl_opts);

  // Builder
  std::unique_ptr<tflite::gpu::InferenceBuilder> builder;
  auto nb = env->NewInferenceBuilder(std::move(graph), gl_opts, &builder);
  if (!nb.ok() || !builder) { LOGE("NewInferenceBuilder failed: %s", nb.message().data()); delete impl; return nullptr; }

  // 输入定义：外部提供 OPENGL_SSBO；布局 DHWC4；数据类型固定为 FLOAT32（与内部一致，避免不支持的转换）
  tg::ObjectDef in_def;
  in_def.data_type = tg::DataType::FLOAT32;
  in_def.data_layout = tg::DataLayout::DHWC4;
  in_def.object_type = tg::ObjectType::OPENGL_SSBO;
  in_def.user_provided = true;

  auto input_defs = builder->inputs();
  if (input_defs.empty()) { LOGE("No inputs in graph"); delete impl; return nullptr; }
  impl->input_w = input_defs[0].dimensions.w;
  impl->input_h = input_defs[0].dimensions.h;
  impl->input_c = input_defs[0].dimensions.c;

  LOGI("SetInputObjectDef: obj=SSBO, layout=DHWC4, dtype=F32, dims=%dx%dx%d",
       impl->input_w, impl->input_h, impl->input_c);
  auto si = builder->SetInputObjectDef(0, in_def);
  if (!si.ok()) { LOGE("SetInputObjectDef failed: %s", si.message().data()); delete impl; return nullptr; }

  // 输出定义：优先使用 OPENGL_SSBO（与编译器默认一致），如需 CPU 读回可在 GetOutput 时转换
  tg::ObjectDef out_def;
  out_def.data_type = tg::DataType::FLOAT32;
  out_def.data_layout = tg::DataLayout::DHWC4;
  out_def.object_type = tg::ObjectType::OPENGL_SSBO;
  out_def.user_provided = true;  // 允许绑定输出 SSBO；若不绑定则内部可能分配

  auto outs = builder->outputs();
  for (int i = 0; i < outs.size(); ++i) {
    auto so = builder->SetOutputObjectDef(i, out_def);
    if (!so.ok()) {
      LOGW("SetOutputObjectDef[%d] as SSBO not supported, fallback CPU_MEMORY", i);
      // 回退为 CPU 内存
      tg::ObjectDef out_cpu;
      out_cpu.data_type = tg::DataType::FLOAT32;
      out_cpu.data_layout = tg::DataLayout::BHWC;
      out_cpu.object_type = tg::ObjectType::CPU_MEMORY;
      out_cpu.user_provided = false;
      auto so2 = builder->SetOutputObjectDef(i, out_cpu);
      if (!so2.ok()) { LOGE("SetOutputObjectDef[%d] CPU failed: %s", i, so2.message().data()); delete impl; return nullptr; }
    }
  }

  // 构建 Runner
  std::unique_ptr<tflite::gpu::InferenceRunner> runner;
  auto br = builder->Build(&runner);
  if (!br.ok() || !runner) { LOGE("Build runner failed: %s", br.message().data()); delete impl; return nullptr; }

  impl->env = std::move(env);
  impl->builder = std::move(builder);
  impl->runner = std::move(runner);

  return static_cast<TfLiteGpuModel>(impl);
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeTexture(TfLiteGpuModel model,
                                                    uint32_t texture_id,
                                                    int width, int height) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner) { LOGE("InvokeTexture: invalid runner"); return false; }
  if (width != impl->input_w || height != impl->input_h) { LOGE("InvokeTexture: size mismatch %dx%d vs %dx%d", width, height, impl->input_w, impl->input_h); return false; }

  // 1) 创建/复用 SSBO
  const int channels = impl->input_c > 0 ? impl->input_c : 4;
  const GLsizei need_bytes = static_cast<GLsizei>(width) * height * 4 * sizeof(float);
  if (impl->input_ssbo == 0 || impl->ssbo_size_bytes != need_bytes) {
    if (impl->input_ssbo != 0) {
      glDeleteBuffers(1, &impl->input_ssbo);
      impl->input_ssbo = 0;
    }
    glGenBuffers(1, &impl->input_ssbo);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, impl->input_ssbo);
    glBufferData(GL_SHADER_STORAGE_BUFFER, need_bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    impl->ssbo_size_bytes = need_bytes;
  }

  // 2) 构建 compute 程序（一次）
  if (impl->convert_program == 0) {
    const char* cs =
        "#version 310 es\n"
        "layout(local_size_x=8, local_size_y=8) in;\n"
        "layout(binding=0, rgba8) uniform readonly lowp image2D srcImg;\n"
        "layout(std430, binding=1) buffer OutSSBO { float data[]; };\n"
        "uniform int width;\n"
        "uniform int height;\n"
        "uniform int channels;\n"
        "void main(){\n"
        "  uvec2 gid = gl_GlobalInvocationID.xy;\n"
        "  if (gid.x>=uint(width) || gid.y>=uint(height)) return;\n"
        "  vec4 pix = imageLoad(srcImg, ivec2(gid));\n"
        "  int x=int(gid.x); int y=int(gid.y);\n"
        "  int base = ((0*height + y)*width + x)*4;\n"
        "  data[base+0] = pix.r;\n"
        "  data[base+1] = (channels>1)?pix.g:0.0;\n"
        "  data[base+2] = (channels>2)?pix.b:0.0;\n"
        "  data[base+3] = (channels>3)?pix.a:0.0;\n"
        "}\n";

    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &cs, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE; glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) { LOGE("Compute shader compile failed"); glDeleteShader(shader); return false; }
    impl->convert_program = glCreateProgram();
    glAttachShader(impl->convert_program, shader);
    glLinkProgram(impl->convert_program);
    glDeleteShader(shader);
    glGetProgramiv(impl->convert_program, GL_LINK_STATUS, &ok);
    if (!ok) { LOGE("Program link failed"); glDeleteProgram(impl->convert_program); impl->convert_program=0; return false; }
  }

  // 3) 绑定纹理为 image0，绑定 SSBO 为 binding=1，调度计算
  glUseProgram(impl->convert_program);
  GLint loc_w = glGetUniformLocation(impl->convert_program, "width");
  GLint loc_h = glGetUniformLocation(impl->convert_program, "height");
  GLint loc_c = glGetUniformLocation(impl->convert_program, "channels");
  if (loc_w>=0) glUniform1i(loc_w, width);
  if (loc_h>=0) glUniform1i(loc_h, height);
  if (loc_c>=0) glUniform1i(loc_c, channels);

  glBindImageTexture(0, static_cast<GLuint>(texture_id), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, impl->input_ssbo);

  const GLuint gx = (width + 7) / 8;
  const GLuint gy = (height + 7) / 8;
  glDispatchCompute(gx, gy, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

  // 4) 将 SSBO 绑定为 InferenceRunner 的输入对象（要求 in_def 为 F32/DHWC4/SSBO）
  tg::OpenGlBuffer buffer; buffer.id = impl->input_ssbo;
  tg::TensorObject in_obj = buffer;
  auto si = impl->runner->SetInputObject(0, in_obj);
  if (!si.ok()) { LOGE("SetInputObject(SSBO) failed: %s", si.message().data()); return false; }

  auto rs = impl->runner->Run();
  if (!rs.ok()) { LOGE("runner->Run failed: %s", rs.message().data()); return false; }
  return true;
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelBindInputSSBO(TfLiteGpuModel model,
                                                    uint32_t ssbo_id) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner) { LOGE("BindInputSSBO: invalid runner"); return false; }
  tg::OpenGlBuffer buffer; buffer.id = static_cast<GLuint>(ssbo_id);
  tg::TensorObject in_obj = buffer;
  auto si = impl->runner->SetInputObject(0, in_obj);
  if (!si.ok()) { LOGE("BindInputSSBO failed: %s", si.message().data()); return false; }
  return true;
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelBindOutputSSBO(TfLiteGpuModel model,
                                                     int index,
                                                     uint32_t ssbo_id) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner) { LOGE("BindOutputSSBO: invalid runner"); return false; }
  tg::OpenGlBuffer buffer; buffer.id = static_cast<GLuint>(ssbo_id);
  tg::TensorObject out_obj = buffer;
  auto so = impl->runner->SetOutputObject(index, out_obj);
  if (!so.ok()) { LOGE("BindOutputSSBO failed: %s", so.message().data()); return false; }
  return true;
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeBuffer(TfLiteGpuModel model,
                                                   const void* input_data,
                                                   int width, int height,
                                                   int channels) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner || !input_data) { LOGE("InvokeBuffer: invalid args"); return false; }

  // 用 CPU_MEMORY 作为输入对象（让 TFLite GPU 执行必要的转换），非零拷贝但兼容 CPU 输入路径。
  tg::ObjectDef in_def;
  in_def.data_type = tg::DataType::FLOAT32;
  in_def.data_layout = tg::DataLayout::BHWC;
  in_def.object_type = tg::ObjectType::CPU_MEMORY;
  in_def.user_provided = true;

  // 检查形状一致
  if (width != impl->input_w || height != impl->input_h) { LOGE("InvokeBuffer: size mismatch %dx%d vs %dx%d", width, height, impl->input_w, impl->input_h); return false; }
  if (channels != impl->input_c && channels != 3 && channels != 4) { LOGE("InvokeBuffer: channels unsupported %d", channels); return false; }

  tg::CpuMemory cpu_mem{const_cast<void*>(input_data),
                        static_cast<size_t>(width * height * channels * sizeof(float))};
  tg::TensorObject obj = cpu_mem;
  auto si = impl->runner->SetInputObject(0, obj);
  if (!si.ok()) { LOGE("SetInputObject(CPU) failed: %s", si.message().data()); return false; }
  auto rs = impl->runner->Run();
  if (!rs.ok()) { LOGE("runner->Run failed: %s", rs.message().data()); return false; }
  return true;
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelGetOutput(TfLiteGpuModel model,
                                                TfLiteOutputs* outputs) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner || !impl->builder || !outputs) { LOGE("GetOutput: invalid args"); return false; }

  auto out_defs = impl->builder->outputs();
  const int num = static_cast<int>(out_defs.size());
  if (outputs->size <= 0 || outputs->outputs == nullptr) { LOGE("GetOutput: outputs buffer missing"); return false; }
  const int cap = outputs->size;
  int filled = 0;
  for (int i = 0; i < num && i < cap; ++i) {
    tg::TensorObject obj;
    auto st = impl->runner->GetOutputObject(i, &obj);
    if (!st.ok()) { LOGE("GetOutputObject[%d] failed: %s", i, st.message().data()); return false; }
    if (!std::holds_alternative<tg::CpuMemory>(obj)) { LOGE("GetOutput: not CPU memory"); return false; }
    const tg::CpuMemory& mem = std::get<tg::CpuMemory>(obj);

    TfLiteOutput& out = outputs->outputs[i];
    out.data = reinterpret_cast<const float*>(mem.data);
    out.size = static_cast<int>(mem.size_bytes / sizeof(float));
    out.height = out_defs[i].dimensions.h;
    out.width  = out_defs[i].dimensions.w;
    out.channels = out_defs[i].dimensions.c;
    ++filled;
  }
  outputs->size = filled;
  if (filled <= 0) { LOGE("GetOutput: no outputs"); return false; }
  return true;
}

TFLITE_WRAP_EXPORT void TfLiteGpuModelDelete(TfLiteGpuModel model) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl) return;
  impl->runner.reset();
  impl->builder.reset();
  impl->env.reset();
  impl->model.reset();
  delete impl;
}

TFLITE_WRAP_EXPORT const char* TfLiteWrapperVersion() {
  return "2.0.0-gpu-gl";
}

}  // extern "C"
