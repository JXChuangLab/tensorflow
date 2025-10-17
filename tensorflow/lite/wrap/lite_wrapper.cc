#include "tensorflow/lite/wrap/lite_wrapper.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

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

  auto* impl = new LiteGpuModelImpl();
  impl->model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!impl->model) { delete impl; return nullptr; }

  // 将 FlatBuffer 模型转换为 GPU Graph
  tg::GraphFloat32 graph;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  auto mb = tg::BuildFromFlatBuffer(*impl->model, resolver, &graph,
                                    /*allow_quant_ops=*/true,
                                    /*apply_model_transformations=*/true);
  if (!mb.ok()) { delete impl; return nullptr; }

  // 创建 GL 环境
  std::unique_ptr<gl::InferenceEnvironment> env;
  gl::InferenceEnvironmentOptions env_opts;
  gl::InferenceEnvironmentProperties props;
  auto env_status = gl::NewInferenceEnvironment(env_opts, &env, &props);
  if (!env_status.ok() || !env) { delete impl; return nullptr; }

  // 选项
  gl::InferenceOptions gl_opts;
  FillOptionsFromPriority(priority, &impl->use_fp16, &gl_opts);

  // Builder
  std::unique_ptr<tflite::gpu::InferenceBuilder> builder;
  auto nb = env->NewInferenceBuilder(std::move(graph), gl_opts, &builder);
  if (!nb.ok() || !builder) { delete impl; return nullptr; }

  // 输入定义：外部提供 OPENGL_TEXTURE；布局 DHWC4；数据类型按精度选择
  tg::ObjectDef in_def;
  in_def.data_type = impl->use_fp16 ? tg::DataType::FLOAT16 : tg::DataType::FLOAT32;
  in_def.data_layout = tg::DataLayout::DHWC4;
  in_def.object_type = tg::ObjectType::OPENGL_TEXTURE;
  in_def.user_provided = true;

  auto input_defs = builder->inputs();
  if (input_defs.empty()) { delete impl; return nullptr; }
  impl->input_w = input_defs[0].dimensions.w;
  impl->input_h = input_defs[0].dimensions.h;
  impl->input_c = input_defs[0].dimensions.c;

  auto si = builder->SetInputObjectDef(0, in_def);
  if (!si.ok()) { delete impl; return nullptr; }

  // 输出定义：使用 CPU_MEMORY，方便 CPU 侧直接读取（MediaPipe 也是 SSBO/CPU 方案）
  tg::ObjectDef out_def;
  out_def.data_type = tg::DataType::FLOAT32;
  out_def.data_layout = tg::DataLayout::BHWC;
  out_def.object_type = tg::ObjectType::CPU_MEMORY;
  out_def.user_provided = false;

  auto outs = builder->outputs();
  for (int i = 0; i < outs.size(); ++i) {
    auto so = builder->SetOutputObjectDef(i, out_def);
    if (!so.ok()) { delete impl; return nullptr; }
  }

  // 构建 Runner
  std::unique_ptr<tflite::gpu::InferenceRunner> runner;
  auto br = builder->Build(&runner);
  if (!br.ok() || !runner) { delete impl; return nullptr; }

  impl->env = std::move(env);
  impl->builder = std::move(builder);
  impl->runner = std::move(runner);

  return static_cast<TfLiteGpuModel>(impl);
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeTexture(TfLiteGpuModel model,
                                                    uint32_t texture_id,
                                                    int width, int height) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner) return false;
  if (width != impl->input_w || height != impl->input_h) {
    // 保持零拷贝：不在内部做 resize。
    return false;
  }

  tg::OpenGlTexture tex{static_cast<GLuint>(texture_id), GL_RGBA8};
  tg::TensorObject in_obj = tex;
  auto si = impl->runner->SetInputObject(0, in_obj);
  if (!si.ok()) return false;

  auto rs = impl->runner->Run();
  return rs.ok();
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelInvokeBuffer(TfLiteGpuModel model,
                                                   const void* input_data,
                                                   int width, int height,
                                                   int channels) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner || !input_data) return false;

  // 用 CPU_MEMORY 作为输入对象（让 TFLite GPU 执行必要的转换），非零拷贝但兼容 CPU 输入路径。
  tg::ObjectDef in_def;
  in_def.data_type = tg::DataType::FLOAT32;
  in_def.data_layout = tg::DataLayout::BHWC;
  in_def.object_type = tg::ObjectType::CPU_MEMORY;
  in_def.user_provided = true;

  // 检查形状一致
  if (width != impl->input_w || height != impl->input_h) {
    return false;
  }
  if (channels != impl->input_c && channels != 3 && channels != 4) {
    return false;
  }

  tg::CpuMemory cpu_mem{const_cast<void*>(input_data),
                        static_cast<size_t>(width * height * channels * sizeof(float))};
  tg::TensorObject obj = cpu_mem;
  auto si = impl->runner->SetInputObject(0, obj);
  if (!si.ok()) return false;
  auto rs = impl->runner->Run();
  return rs.ok();
}

TFLITE_WRAP_EXPORT bool TfLiteGpuModelGetOutput(TfLiteGpuModel model,
                                                TfLiteOutputs* outputs) {
  auto* impl = static_cast<LiteGpuModelImpl*>(model);
  if (!impl || !impl->runner || !impl->builder || !outputs) return false;

  auto out_defs = impl->builder->outputs();
  const int num = static_cast<int>(out_defs.size());
  if (outputs->size <= 0 || outputs->outputs == nullptr) return false;
  const int cap = outputs->size;
  int filled = 0;
  for (int i = 0; i < num && i < cap; ++i) {
    tg::TensorObject obj;
    auto st = impl->runner->GetOutputObject(i, &obj);
    if (!st.ok()) return false;
    if (!std::holds_alternative<tg::CpuMemory>(obj)) return false;
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
  return filled > 0;
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
