#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"
#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/litert-micro/memory_util.h"
#include "sw/opt/rvv_opt.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_partial_layers.h"

namespace {

using MobilenetOpResolver = tflite::MicroMutableOpResolver<2>;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;

constexpr uintptr_t kExtmemBase = 0x20000000;
constexpr size_t kPartialTensorArenaSize = 768 * 1024;

volatile opennpux_coral_mobilenet_mailbox* Mailbox() {
  return reinterpret_cast<volatile opennpux_coral_mobilenet_mailbox*>(
      kExtmemBase + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET);
}

coral_operator_descriptor* ExecutionDescriptor() {
  return reinterpret_cast<coral_operator_descriptor*>(
      kExtmemBase + CORAL_EXECUTION_DESCRIPTOR_OFFSET);
}

coral_operator_descriptor* OperatorDescriptor() {
  return reinterpret_cast<coral_operator_descriptor*>(
      kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET);
}

void MarkProgress(uint32_t value) {
  *reinterpret_cast<volatile uint32_t*>(
      OPENNPUX_CORAL_MOBILENET_PROGRESS_ADDR) = value;
  asm volatile("fence rw, rw" ::: "memory");
}

TfLiteStatus (*convInvoke)(TfLiteContext*, TfLiteNode*) = nullptr;
TfLiteStatus (*depthwiseInvoke)(TfLiteContext*, TfLiteNode*) = nullptr;

uint32_t TensorBytes(const TfLiteEvalTensor* tensor, uint32_t element_size) {
  if (tensor == nullptr || tensor->dims == nullptr || tensor->dims->size <= 0 ||
      tensor->dims->size > CORAL_OPERATOR_MAX_DIMS) {
    return 0;
  }
  uint64_t bytes = element_size;
  for (int i = 0; i < tensor->dims->size; ++i) {
    if (tensor->dims->data[i] <= 0 ||
        bytes > UINT32_MAX / static_cast<uint32_t>(tensor->dims->data[i])) {
      return 0;
    }
    bytes *= tensor->dims->data[i];
  }
  return static_cast<uint32_t>(bytes);
}

void TensorDimensions(
    const TfLiteEvalTensor* tensor,
    uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS]) {
  for (size_t i = 0; i < CORAL_OPERATOR_MAX_DIMS; ++i) dimensions[i] = 0;
  for (int i = 0; i < tensor->dims->size; ++i) {
    dimensions[i] = tensor->dims->data[i];
  }
}

bool StageTensor(
    opennpux::OperatorStagingAllocator* allocator,
    coral_operator_descriptor* descriptor, uint32_t index,
    const TfLiteEvalTensor* tensor, uint32_t element_size,
    uint32_t element_type, int32_t zero_point) {
  const uint32_t bytes = TensorBytes(tensor, element_size);
  const uint32_t address = opennpux::AllocateStaging(
      allocator, bytes, element_size < 16 ? 16 : element_size);
  if (address == 0) return false;
  uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS] = {};
  TensorDimensions(tensor, dimensions);
  if (!opennpux::SetOperatorTensor(
          descriptor, index, address, bytes, tensor->dims->size, dimensions,
          element_type, zero_point)) {
    return false;
  }
  coralnpu_v2::opt::Memcpy(
      reinterpret_cast<void*>(address), tensor->data.raw, bytes);
  return true;
}

bool TryHybridConvInvoke(
    TfLiteContext* context, TfLiteNode* node, bool depthwise,
    TfLiteStatus* status) {
  const uint32_t opcode = depthwise ?
      CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8 :
      CORAL_OPERATOR_OP_CONV_2D_INT8;
  if (opennpux::OperatorMode() != CORAL_OPERATOR_MODE_HYBRID ||
      !opennpux::HybridOperatorSupported(opcode) || node == nullptr ||
      node->user_data == nullptr || node->builtin_data == nullptr ||
      tflite::NumInputs(node) != 3) {
    return false;
  }

  const int input_index = depthwise ? tflite::kDepthwiseConvInputTensor :
                                      tflite::kConvInputTensor;
  const int filter_index = depthwise ? tflite::kDepthwiseConvWeightsTensor :
                                       tflite::kConvWeightsTensor;
  const int bias_index = depthwise ? tflite::kDepthwiseConvBiasTensor :
                                     tflite::kConvBiasTensor;
  const int output_index = depthwise ? tflite::kDepthwiseConvOutputTensor :
                                       tflite::kConvOutputTensor;
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, input_index);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, filter_index);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, bias_index);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, output_index);
  if (input == nullptr || filter == nullptr || bias == nullptr ||
      output == nullptr || input->type != kTfLiteInt8 ||
      filter->type != kTfLiteInt8 || bias->type != kTfLiteInt32 ||
      output->type != kTfLiteInt8) {
    return false;
  }

  const auto& data = *static_cast<const
      coralnpu_v2::opt::litert_micro::OpDataConvCustom*>(node->user_data);
  const uint32_t output_channels = output->dims->data[3];
  opennpux::OperatorStagingAllocator allocator = {};
  opennpux::InitializeStagingAllocator(&allocator, kExtmemBase);
  coral_operator_descriptor* descriptor = OperatorDescriptor();
  opennpux::InitializeOperatorDescriptor(
      descriptor, opcode, CORAL_OPERATOR_MODE_HYBRID);
  if (!StageTensor(&allocator, descriptor, 0, input, 1,
                   CORAL_OPERATOR_ELEMENT_INT8, data.input_zero_point) ||
      !StageTensor(&allocator, descriptor, 1, filter, 1,
                   CORAL_OPERATOR_ELEMENT_INT8, data.filter_zero_point) ||
      !StageTensor(&allocator, descriptor, 2, bias, sizeof(int32_t),
                   CORAL_OPERATOR_ELEMENT_INT32, 0) ||
      !StageTensor(&allocator, descriptor, 3, output, 1,
                   CORAL_OPERATOR_ELEMENT_INT8, data.output_zero_point)) {
    return false;
  }

  descriptor->multiplier_address = opennpux::AllocateStaging(
      &allocator, output_channels * sizeof(int32_t), alignof(int32_t));
  descriptor->shift_address = opennpux::AllocateStaging(
      &allocator, output_channels * sizeof(int32_t), alignof(int32_t));
  if (descriptor->multiplier_address == 0 || descriptor->shift_address == 0) {
    return false;
  }
  coralnpu_v2::opt::Memcpy(
      reinterpret_cast<void*>(descriptor->multiplier_address),
      data.per_channel_output_multiplier,
      output_channels * sizeof(int32_t));
  coralnpu_v2::opt::Memcpy(
      reinterpret_cast<void*>(descriptor->shift_address),
      data.per_channel_output_shift, output_channels * sizeof(int32_t));
  descriptor->quantization_count = output_channels;
  descriptor->output_zero_point = data.output_zero_point;
  descriptor->activation_min = data.output_activation_min;
  descriptor->activation_max = data.output_activation_max;
  if (depthwise) {
    const auto& params = *reinterpret_cast<const TfLiteDepthwiseConvParams*>(
        node->builtin_data);
    descriptor->stride_height = params.stride_height;
    descriptor->stride_width = params.stride_width;
  } else {
    const auto& params = *reinterpret_cast<const TfLiteConvParams*>(
        node->builtin_data);
    descriptor->stride_height = params.stride_height;
    descriptor->stride_width = params.stride_width;
  }
  descriptor->padding_height = data.padding.height;
  descriptor->padding_width = data.padding.width;

  if (!opennpux::SubmitHybridOperator(
          descriptor, kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET)) {
    *status = kTfLiteError;
    return true;
  }
  coralnpu_v2::opt::Memcpy(
      output->data.raw,
      reinterpret_cast<const void*>(descriptor->tensors[3].address),
      descriptor->tensors[3].size);
  coral_operator_descriptor* execution = ExecutionDescriptor();
  execution->operation_count += descriptor->operation_count;
  execution->host_elapsed_ns += descriptor->host_elapsed_ns;
  execution->modeled_cycles += descriptor->modeled_cycles;
  execution->bytes_read += descriptor->bytes_read;
  execution->bytes_written += descriptor->bytes_written;
  *status = kTfLiteOk;
  return true;
}

TfLiteStatus TracedConvInvoke(TfLiteContext* context, TfLiteNode* node) {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_BEGIN);
  TfLiteStatus status = kTfLiteError;
  if (!TryHybridConvInvoke(context, node, false, &status)) {
    status = convInvoke(context, node);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_END);
  return status;
}

TfLiteStatus TracedDepthwiseInvoke(TfLiteContext* context, TfLiteNode* node) {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_DEPTHWISE_BEGIN);
  TfLiteStatus status = kTfLiteError;
  if (!TryHybridConvInvoke(context, node, true, &status)) {
    status = depthwiseInvoke(context, node);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_DEPTHWISE_END);
  return status;
}

TfLiteStatus RegisterOps(MobilenetOpResolver& resolver) {
  TFLMRegistration conv = Register_CONV_2D();
  convInvoke = conv.invoke;
  conv.invoke = TracedConvInvoke;
  TF_LITE_ENSURE_STATUS(resolver.AddConv2D(conv));

  TFLMRegistration depthwise = Register_DEPTHWISE_CONV_2D();
  depthwiseInvoke = depthwise.invoke;
  depthwise.invoke = TracedDepthwiseInvoke;
  TF_LITE_ENSURE_STATUS(resolver.AddDepthwiseConv2D(depthwise));
  return kTfLiteOk;
}

__attribute__((constructor(101))) void MarkCrtReady() {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_CRT);
}

int Fail(uint32_t error) {
  Mailbox()->error_code = error;
  Mailbox()->state = OPENNPUX_CORAL_MOBILENET_ERROR;
  asm volatile("fence rw, rw" ::: "memory");
  return -1;
}

}  // namespace

extern "C" {

__attribute__((used)) uint32_t opennpux_mobilenet_early_progress_enabled = 1;

[[noreturn]] void coralnpu_exception_handler() {
  uint32_t mepc;
  uint32_t mtval;
  uint32_t mcause;
  asm volatile("csrr %0, mepc" : "=r"(mepc));
  asm volatile("csrr %0, mtval" : "=r"(mtval));
  asm volatile("csrr %0, mcause" : "=r"(mcause));
  MarkProgress(UINT32_C(0x4d4eff01));
  MarkProgress(mepc);
  MarkProgress(UINT32_C(0x4d4eff02));
  MarkProgress(mtval);
  MarkProgress(UINT32_C(0x4d4eff03));
  MarkProgress(mcause);
  asm volatile("ebreak");
  while (true) {
  }
}

uint8_t tensor_arena[kPartialTensorArenaSize]
    __attribute__((section(".bss"), aligned(16)));

}  // extern "C"

int main() {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_MAIN);
  volatile opennpux_coral_mobilenet_mailbox* mailbox = Mailbox();
  mailbox->magic = OPENNPUX_CORAL_MOBILENET_MAGIC;
  mailbox->version = OPENNPUX_CORAL_MOBILENET_VERSION;
  mailbox->state = OPENNPUX_CORAL_MOBILENET_STARTED;
  mailbox->error_code = OPENNPUX_CORAL_MOBILENET_ERROR_NONE;
  mailbox->output_count = 0;
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_MAILBOX);

  const tflite::Model* model =
      tflite::GetModel(g_mobilenet_v1_025_partial_layers_model_data);
  MobilenetOpResolver resolver;
  if (RegisterOps(resolver) != kTfLiteOk) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_ALLOCATE);
  }

  tflite::MicroInterpreter interpreter(
      model, resolver, tensor_arena, sizeof(tensor_arena));
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_ALLOCATE_BEGIN);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_ALLOCATE);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_ALLOCATE_END);
  TfLiteTensor* input = interpreter.input(0);
  if (input == nullptr || input->data.data == nullptr || input->bytes == 0) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INPUT);
  }
  coralnpu_v2::opt::Memset(input->data.data, 0, input->bytes);
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INPUT_READY);
  mailbox->state = OPENNPUX_CORAL_MOBILENET_TENSORS_READY;

  const uint64_t start_cycles = mcycle_read();
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_BEGIN);
  const uint32_t execution_mode = opennpux::OperatorMode();
  coral_operator_descriptor* descriptor = ExecutionDescriptor();
  opennpux::InitializeOperatorDescriptor(
      descriptor, CORAL_OPERATOR_OP_PARTIAL_MOBILENET, execution_mode);
  descriptor->flags = CORAL_OPERATOR_FLAG_ALLOW_RTL_FALLBACK;

  descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
  opennpux::OperatorFence();
  if (interpreter.Invoke() != kTfLiteOk) {
    descriptor->state = CORAL_OPERATOR_STATE_ERROR;
    descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INVOKE);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END);
  const uint64_t elapsed_cycles = mcycle_read() - start_cycles;
  if (execution_mode == CORAL_OPERATOR_MODE_RTL) {
    descriptor->modeled_cycles = elapsed_cycles;
  }

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr ||
      output->bytes < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT) {
    descriptor->state = CORAL_OPERATOR_STATE_ERROR;
    descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_OUTPUT);
  }
  for (uint32_t i = 0; i < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT; ++i) {
    mailbox->output[i] = output->data.int8[i];
  }
  mailbox->cycle_low = static_cast<uint32_t>(descriptor->modeled_cycles);
  mailbox->cycle_high =
      static_cast<uint32_t>(descriptor->modeled_cycles >> 32);
  mailbox->output_count = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  mailbox->state = OPENNPUX_CORAL_MOBILENET_COMPLETE;
  if (execution_mode == CORAL_OPERATOR_MODE_RTL) {
    descriptor->bytes_written = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  }
  descriptor->state = CORAL_OPERATOR_STATE_COMPLETE;
  descriptor->error = CORAL_OPERATOR_ERROR_NONE;
  opennpux::OperatorFence();
  return 0;
}
