#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"
#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/rvv_opt.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
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

TfLiteStatus TracedConvInvoke(TfLiteContext* context, TfLiteNode* node) {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_BEGIN);
  const TfLiteStatus status = convInvoke(context, node);
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_END);
  return status;
}

TfLiteStatus TracedDepthwiseInvoke(TfLiteContext* context, TfLiteNode* node) {
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_DEPTHWISE_BEGIN);
  const TfLiteStatus status = depthwiseInvoke(context, node);
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
  uint32_t execution_mode = opennpux::OperatorMode();
  if (execution_mode == CORAL_OPERATOR_MODE_HYBRID &&
      !opennpux::HybridOperatorSupported(
          CORAL_OPERATOR_OP_PARTIAL_MOBILENET)) {
    execution_mode = CORAL_OPERATOR_MODE_RTL;
  }
  coral_operator_descriptor* descriptor = OperatorDescriptor();
  opennpux::InitializeOperatorDescriptor(
      descriptor, CORAL_OPERATOR_OP_PARTIAL_MOBILENET, execution_mode);
  descriptor->flags = CORAL_OPERATOR_FLAG_ALLOW_RTL_FALLBACK;

  if (execution_mode == CORAL_OPERATOR_MODE_HYBRID) {
    if (!opennpux::SubmitHybridOperator(
            descriptor, kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET)) {
      return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INVOKE);
    }
    MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END);
    return 0;
  }
  descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
  opennpux::OperatorFence();
  if (interpreter.Invoke() != kTfLiteOk) {
    descriptor->state = CORAL_OPERATOR_STATE_ERROR;
    descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INVOKE);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END);
  const uint64_t elapsed_cycles = mcycle_read() - start_cycles;
  descriptor->modeled_cycles = elapsed_cycles;

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
  mailbox->cycle_low = static_cast<uint32_t>(elapsed_cycles);
  mailbox->cycle_high = static_cast<uint32_t>(elapsed_cycles >> 32);
  mailbox->output_count = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  mailbox->state = OPENNPUX_CORAL_MOBILENET_COMPLETE;
  descriptor->bytes_written = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  descriptor->state = CORAL_OPERATOR_STATE_COMPLETE;
  descriptor->error = CORAL_OPERATOR_ERROR_NONE;
  opennpux::OperatorFence();
  return 0;
}
