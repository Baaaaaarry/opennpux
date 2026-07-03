#include <cstddef>
#include <cstdint>
#include <cstring>

#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "sw/opt/litert-micro/conv.h"
#include "sw/opt/litert-micro/depthwise_conv.h"
#include "sw/opt/rvv_opt.h"
#include "sw/utils/utils.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tests/cocotb/tutorial/tfmicro/mobilenet_v1_025_224_int8_dummy.h"

namespace {

using MobilenetOpResolver = tflite::MicroMutableOpResolver<10>;
using coralnpu_v2::opt::litert_micro::Register_CONV_2D;
using coralnpu_v2::opt::litert_micro::Register_DEPTHWISE_CONV_2D;

constexpr uintptr_t kExtmemBase = 0x20000000;

TfLiteStatus RegisterOps(MobilenetOpResolver& resolver) {
  TF_LITE_ENSURE_STATUS(resolver.AddConv2D(Register_CONV_2D()));
  TF_LITE_ENSURE_STATUS(
      resolver.AddDepthwiseConv2D(Register_DEPTHWISE_CONV_2D()));
  TF_LITE_ENSURE_STATUS(resolver.AddReshape());
  TF_LITE_ENSURE_STATUS(resolver.AddAveragePool2D());
  TF_LITE_ENSURE_STATUS(resolver.AddSoftmax());
  TF_LITE_ENSURE_STATUS(resolver.AddStridedSlice());
  TF_LITE_ENSURE_STATUS(resolver.AddPad());
  TF_LITE_ENSURE_STATUS(resolver.AddMean());
  TF_LITE_ENSURE_STATUS(resolver.AddShape());
  TF_LITE_ENSURE_STATUS(resolver.AddPack());
  return kTfLiteOk;
}

volatile opennpux_coral_mobilenet_mailbox* Mailbox() {
  return reinterpret_cast<volatile opennpux_coral_mobilenet_mailbox*>(
      kExtmemBase + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET);
}

void MarkProgress(uint32_t value) {
  *reinterpret_cast<volatile uint32_t*>(
      OPENNPUX_CORAL_MOBILENET_PROGRESS_ADDR) = value;
  asm volatile("fence rw, rw" ::: "memory");
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

uint8_t tensor_arena[OPENNPUX_CORAL_MOBILENET_ARENA_SIZE]
    __attribute__((section(".extbss"), aligned(16)));
uint8_t inference_input[224 * 224 * 3]
    __attribute__((section(".data"), aligned(16)));

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
      tflite::GetModel(g_25_224_int8_dummy_model_data);
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
  if (input == nullptr || input->bytes != sizeof(inference_input)) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INPUT);
  }
  coralnpu_v2::opt::Memcpy(
      input->data.data, inference_input, sizeof(inference_input));
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INPUT_READY);
  mailbox->state = OPENNPUX_CORAL_MOBILENET_TENSORS_READY;

  const uint64_t start_cycles = mcycle_read();
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_BEGIN);
  if (interpreter.Invoke() != kTfLiteOk) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_INVOKE);
  }
  MarkProgress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END);
  const uint64_t elapsed_cycles = mcycle_read() - start_cycles;

  TfLiteTensor* output = interpreter.output(0);
  if (output == nullptr ||
      output->bytes < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT) {
    return Fail(OPENNPUX_CORAL_MOBILENET_ERROR_OUTPUT);
  }
  for (uint32_t i = 0; i < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT; ++i) {
    mailbox->output[i] = output->data.int8[i];
  }
  mailbox->cycle_low = static_cast<uint32_t>(elapsed_cycles);
  mailbox->cycle_high = static_cast<uint32_t>(elapsed_cycles >> 32);
  mailbox->output_count = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  mailbox->state = OPENNPUX_CORAL_MOBILENET_COMPLETE;
  asm volatile("fence rw, rw" ::: "memory");
  return 0;
}
