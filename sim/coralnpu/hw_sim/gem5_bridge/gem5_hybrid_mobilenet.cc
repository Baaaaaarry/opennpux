#include "hw_sim/gem5_bridge/gem5_hybrid_mobilenet.h"

#include <chrono>
#include <cstring>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_partial_layers.h"

bool RunGem5HybridMobilenet(int32_t output[5], uint64_t* elapsed_ns) {
  constexpr size_t kArenaSize = 2 * 1024 * 1024;
  std::vector<uint8_t> arena(kArenaSize, 0);
  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_partial_layers_model_data);
  tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return false;
  }

  tflite::MicroInterpreter interpreter(model, resolver, arena.data(),
                                       arena.size());
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return false;
  }
  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr || input->data.data == nullptr || input->bytes == 0) {
    return false;
  }
  std::memset(input->data.data, 0, input->bytes);

  const auto start = std::chrono::steady_clock::now();
  if (interpreter.Invoke() != kTfLiteOk) {
    return false;
  }
  const auto end = std::chrono::steady_clock::now();
  TfLiteTensor* result = interpreter.output(0);
  if (result == nullptr || result->bytes < 5) {
    return false;
  }
  for (size_t i = 0; i < 5; ++i) {
    output[i] = result->data.int8[i];
  }
  *elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
  return true;
}
