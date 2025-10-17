#include "tensorflow/lite/wrap/lite_gpu_runner.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/lite/delegates/gpu/common/model_builder.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/api.h"
#include "tensorflow/lite/delegates/gpu/gl/api2.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

// GL types are provided by portable_gl31 via api.h

namespace tg = tflite::gpu;
namespace gl = tflite::gpu::gl;

struct LiteGpuRunnerImpl {
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<gl::InferenceEnvironment> env;  // api2.h 中定义的接口类
  std::unique_ptr<tflite::gpu::InferenceBuilder> builder;
  std::unique_ptr<tg::InferenceRunner> runner;  // 来自 api.h（同命名空间）

  tg::TensorObjectDef input_def;
  std::vector<tg::TensorObjectDef> output_defs;

  int input_w = 0;
  int input_h = 0;
  int input_c = 4;  // 默认 RGBA
};

static void FillGlOptions(TfLiteGpuPriority pri, gl::InferenceOptions* opt) {
  opt->usage = tg::InferenceUsage::SUSTAINED_SPEED;
  switch (pri) {
    case TFLITE_GPU_PRIORITY_MAX_PRECISION:
      opt->priority1 = tg::InferencePriority::MAX_PRECISION;
      opt->priority2 = tg::InferencePriority::MIN_LATENCY;
      opt->priority3 = tg::InferencePriority::MIN_MEMORY_USAGE;
      break;
    case TFLITE_GPU_PRIORITY_MIN_MEMORY_USAGE:
      opt->priority1 = tg::InferencePriority::MIN_MEMORY_USAGE;
      opt->priority2 = tg::InferencePriority::MAX_PRECISION;
      opt->priority3 = tg::InferencePriority::MIN_LATENCY;
      break;
    case TFLITE_GPU_PRIORITY_MIN_LATENCY:
    case TFLITE_GPU_PRIORITY_AUTO:
    default:
      opt->priority1 = tg::InferencePriority::MIN_LATENCY;
      opt->priority2 = tg::InferencePriority::MIN_MEMORY_USAGE;
      opt->priority3 = tg::InferencePriority::MAX_PRECISION;
      break;
  }
}

extern "C" {

TfLiteGpuRunner TfLiteGpuRunnerCreate(const char* model_path,
                                      TfLiteGpuPriority priority,
                                      int prefer_fp16) {
  if (!model_path) return nullptr;

  auto* impl = new LiteGpuRunnerImpl();
  impl->model = tflite::FlatBufferModel::BuildFromFile(model_path);
  if (!impl->model) { delete impl; return nullptr; }

  // 将 flatbuffer 模型转换为 GPU Graph
  tg::GraphFloat32 graph;
  // 需要一个 OpResolver；此处使用空 resolver 让 BuildFromFlatBuffer 走默认解析
  tflite::ops::builtin::BuiltinOpResolver resolver;  // 需要包含头：kernels/register.h
  auto status = tg::BuildFromFlatBuffer(*impl->model, resolver, &graph,
                                        /*allow_quant_ops=*/true,
                                        /*apply_model_transformations=*/true);
  if (!status.ok()) { delete impl; return nullptr; }

  // 创建 OpenGL 环境（api2）
  std::unique_ptr<gl::InferenceEnvironment> env;
  gl::InferenceEnvironmentProperties props;
  gl::InferenceEnvironmentOptions env_opts;  // 可选设置 queue
  auto env_status = gl::NewInferenceEnvironment(env_opts, &env, &props);
  if (!env_status.ok() || !env) { delete impl; return nullptr; }

  // 创建 Builder
  gl::InferenceOptions gl_opts;
  FillGlOptions(priority, &gl_opts);
  std::unique_ptr<tflite::gpu::InferenceBuilder> builder;
  auto nb = env->NewInferenceBuilder(std::move(graph), gl_opts, &builder);
  if (!nb.ok() || !builder) { delete impl; return nullptr; }

  // 设定输入/输出对象定义：期望外部提供 OPENGL_TEXTURE，数据类型 FP16/FP32
  // 注意：GL 后端内部一般采用 DHWC4（即 C 对齐 4），外部可提供纹理，runtime 做必要转换
  tg::ObjectDef input_obj_def;
  input_obj_def.data_type = prefer_fp16 ? tg::DataType::FLOAT16 : tg::DataType::FLOAT32;
  input_obj_def.data_layout = tg::DataLayout::DHWC4;  // 贴近内部对齐布局
  input_obj_def.object_type = tg::ObjectType::OPENGL_TEXTURE;
  input_obj_def.user_provided = true;
  // 从 builder 查询输入维度
  auto input_defs = builder->inputs();
  if (input_defs.empty()) { delete impl; return nullptr; }
  impl->input_w = input_defs[0].dimensions.w;
  impl->input_h = input_defs[0].dimensions.h;
  impl->input_c = input_defs[0].dimensions.c;  // 通常为4对齐

  builder->SetInputObjectDef(0, input_obj_def);

  // 输出：默认仍在 GPU 内（纹理），调用时可替换为用户提供输出纹理
  tg::ObjectDef output_obj_def;
  output_obj_def.data_type = prefer_fp16 ? tg::DataType::FLOAT16 : tg::DataType::FLOAT32;
  output_obj_def.data_layout = tg::DataLayout::DHWC4;
  output_obj_def.object_type = tg::ObjectType::OPENGL_TEXTURE;
  output_obj_def.user_provided = true;  // 允许外部绑定输出纹理
  auto output_defs = builder->outputs();
  for (int i = 0; i < output_defs.size(); ++i) {
    builder->SetOutputObjectDef(i, output_obj_def);
  }

  // 构建 Runner
  std::unique_ptr<tg::InferenceRunner> runner;
  if (!builder->Build(&runner).ok() || !runner) { delete impl; return nullptr; }

  impl->env = std::move(env);
  impl->builder = std::move(builder);
  impl->runner = std::move(runner);

  return static_cast<TfLiteGpuRunner>(impl);
}

bool TfLiteGpuRunnerInvokeTexture(TfLiteGpuRunner runner,
                                  uint32_t input_texture_2d,
                                  int tex_w, int tex_h,
                                  uint32_t output_texture_2d) {
  auto* impl = static_cast<LiteGpuRunnerImpl*>(runner);
  if (!impl || !impl->runner) return false;
  if (tex_w != impl->input_w || tex_h != impl->input_h) {
    // 保持零拷贝：不做隐式 resize。请在外部用 FBO+shader 做尺寸匹配。
    return false;
  }

  // 设置输入纹理对象
  tg::OpenGlTexture in_tex{static_cast<GLuint>(input_texture_2d), GL_RGBA8};
  tg::TensorObject in_obj = in_tex;
  auto si = impl->runner->SetInputObject(0, in_obj);
  if (!si.ok()) return false;

  // 如用户提供输出纹理，将其绑定；否则由内部临时对象承接并丢弃
  if (output_texture_2d != 0) {
    tg::OpenGlTexture out_tex{static_cast<GLuint>(output_texture_2d), GL_RGBA8};
    tg::TensorObject out_obj = out_tex;
    auto so = impl->runner->SetOutputObject(0, out_obj);
    if (!so.ok()) return false;
  }

  auto rs = impl->runner->Run();
  return rs.ok();
}

bool TfLiteGpuRunnerReadOutput(TfLiteGpuRunner runner,
                               int index, float* out, int out_count) {
  // 可选：按需实现 GPU->CPU 读回（绑定输出为 CPU_MEMORY），这里先返回 false。
  (void)runner; (void)index; (void)out; (void)out_count;
  return false;
}

int TfLiteGpuRunnerGetNumOutputs(TfLiteGpuRunner runner) {
  auto* impl = static_cast<LiteGpuRunnerImpl*>(runner);
  if (!impl || !impl->builder) return 0;
  return static_cast<int>(impl->builder->outputs().size());
}

void TfLiteGpuRunnerDelete(TfLiteGpuRunner runner) {
  auto* impl = static_cast<LiteGpuRunnerImpl*>(runner);
  if (!impl) return;
  impl->runner.reset();
  impl->builder.reset();
  impl->env.reset();
  impl->model.reset();
  delete impl;
}

} // extern "C"
