#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <cassert>
#include <cmath>
#include <cstring>

namespace {

opennpux_npu_operator_parameters Parameters() {
  opennpux_npu_operator_parameters parameters = {};
  parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters.struct_size = sizeof(parameters);
  parameters.opcode = 2;
  parameters.flags = 1;
  parameters.input_features = 2;
  parameters.output_features = 1;
  parameters.quantization_bits = 4;
  parameters.quantization_group_size = 2;
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  parameters.quantized_zero_bias = 0;
  return parameters;
}

Gem5GenericGptqWeights IdentityWeights(const uint32_t* qweight,
                                       const uint32_t* qzeros,
                                       const float* scales) {
  return {{qweight, sizeof(uint32_t) * 2}, {qzeros, sizeof(uint32_t)},
          {scales, sizeof(float) * 2}, {nullptr, 0}};
}

struct StreamedWeights {
  const void* components[4];
  size_t sizes[4];
};

bool ReadStreamedWeight(void* opaque, Gem5GptqComponent component,
                        uint64_t offset, void* destination, size_t size) {
  auto* weights = static_cast<StreamedWeights*>(opaque);
  const size_t index = static_cast<size_t>(component);
  if (index >= 4 || offset > weights->sizes[index] ||
      size > weights->sizes[index] - offset) {
    return false;
  }
  std::memcpy(destination,
              static_cast<const uint8_t*>(weights->components[index]) + offset,
              size);
  return true;
}

struct RoutedWeights {
  Gem5GenericGptqWeights experts[2][3];
};

bool ProvideRoutedWeights(void* opaque, uint64_t expert_id,
                          Gem5GenericGptqWeights* gate,
                          Gem5GenericGptqWeights* up,
                          Gem5GenericGptqWeights* down) {
  auto* weights = static_cast<RoutedWeights*>(opaque);
  if (expert_id >= 2 || gate == nullptr || up == nullptr || down == nullptr) {
    return false;
  }
  *gate = weights->experts[expert_id][0];
  *up = weights->experts[expert_id][1];
  *down = weights->experts[expert_id][2];
  return true;
}

}  // namespace

int main() {
  const float input[] = {2.0f, 3.0f};
  const uint32_t qweight[] = {UINT32_C(0x21)};
  const uint32_t qzeros[] = {0};
  const float scales[] = {0.5f};
  float output[] = {0.0f};
  const Gem5GenericGptqOperands operands = {
      {input, sizeof(input)},       {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},    {scales, sizeof(scales)},
      {nullptr, 0},                {output, sizeof(output)},
  };
  Gem5GptqKernelStats stats = {};
  auto parameters = Parameters();
  assert(RunGem5GenericGptqMatMul(parameters, 1, operands, &stats));
  assert(output[0] == 4.0f);
  assert(stats.operations == 4);

  parameters.output_features = 8;
  const uint32_t streamed_qweight[8] = {
      UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21),
      UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21)};
  const uint32_t streamed_qzeros[1] = {};
  const float streamed_scales[8] = {
      0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  float streamed_output[8] = {};
  StreamedWeights streamed_weights = {
      {streamed_qweight, streamed_qzeros, streamed_scales, nullptr},
      {sizeof(streamed_qweight), sizeof(streamed_qzeros),
       sizeof(streamed_scales), 0}};
  assert(RunGem5GenericGptqMatMulStreamed(
      parameters, 1, {input, sizeof(input)}, ReadStreamedWeight,
      &streamed_weights, 8, false,
      {streamed_output, sizeof(streamed_output)}, &stats));
  for (float value : streamed_output) {
    assert(value == 4.0f);
  }
  const auto* qweight_bytes =
      reinterpret_cast<const uint8_t*>(streamed_qweight);
  const Gem5GenericGptqPageSpan page_spans[] = {
      {kGem5GptqQweight, 0, qweight_bytes, 16},
      {kGem5GptqQweight, 16, qweight_bytes + 16,
       sizeof(streamed_qweight) - 16},
      {kGem5GptqQzeros, 0, streamed_qzeros, sizeof(streamed_qzeros)},
      {kGem5GptqScales, 0, streamed_scales, sizeof(streamed_scales)},
  };
  Gem5GenericGptqPageReader page_reader = {
      page_spans, sizeof(page_spans) / sizeof(page_spans[0])};
  std::memset(streamed_output, 0, sizeof(streamed_output));
  assert(RunGem5GenericGptqMatMulStreamed(
      parameters, 1, {input, sizeof(input)},
      ReadGem5GenericGptqPageSpans, &page_reader, 8, false,
      {streamed_output, sizeof(streamed_output)}, &stats));
  for (float value : streamed_output) {
    assert(value == 4.0f);
  }
  page_reader.span_count = 1;
  assert(!RunGem5GenericGptqMatMulStreamed(
      parameters, 1, {input, sizeof(input)},
      ReadGem5GenericGptqPageSpans, &page_reader, 8, false,
      {streamed_output, sizeof(streamed_output)}, &stats));

  uint8_t page_cache[32];
  for (size_t index = 0; index < sizeof(page_cache); ++index) {
    page_cache[index] = static_cast<uint8_t>(index);
  }
  opennpux_npu_page_fault fault = {};
  fault.file_offset = 0x1000;
  fault.page_size = 16;
  fault.range_file_offset = 0x1008;
  fault.range_size = 12;
  fault.component_id = OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT;
  fault.cache_slot = 1;
  Gem5GenericGptqPageSpan built_span = {};
  assert(BuildGem5GenericGptqPageSpan(
      fault, page_cache, sizeof(page_cache), &built_span));
  assert(built_span.component == kGem5GptqQweight);
  assert(built_span.tensor_offset == 0 && built_span.size == 8);
  assert(built_span.data == page_cache + 24);
  fault.file_offset = 0x1010;
  fault.cache_slot = 0;
  assert(BuildGem5GenericGptqPageSpan(
      fault, page_cache, sizeof(page_cache), &built_span));
  assert(built_span.tensor_offset == 8 && built_span.size == 4);
  assert(built_span.data == page_cache);
  fault.component_id = 0;
  assert(!BuildGem5GenericGptqPageSpan(
      fault, page_cache, sizeof(page_cache), &built_span));

  struct {
    opennpux_npu_weight_residency_header header;
    opennpux_npu_weight_residency_record records[2];
  } residency = {};
  residency.header.magic = OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC;
  residency.header.version = OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION;
  residency.header.header_size = sizeof(residency.header);
  residency.header.record_size = sizeof(residency.records[0]);
  residency.header.capacity = 2;
  residency.header.valid_records = 2;
  for (uint32_t index = 0; index < 2; ++index) {
    residency.records[index].command_id = 9;
    residency.records[index].role_id = 10;
    residency.records[index].component_id =
        OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT;
    residency.records[index].expert_id = 11;
    residency.records[index].range_file_offset = 0x1008;
    residency.records[index].range_size = 24;
    residency.records[index].page_file_offset = 0x1000 + index * 16;
    residency.records[index].cache_slot = index;
    residency.records[index].page_size = 16;
    residency.records[index].flags = OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID;
  }
  Gem5GenericGptqPageSpan resident_spans[2] = {};
  size_t resident_span_count = 0;
  assert(BuildGem5GenericGptqResidentSpans(
      &residency.header, sizeof(residency), 9, 10, 11, page_cache,
      sizeof(page_cache), resident_spans, 2, &resident_span_count));
  assert(resident_span_count == 2);
  assert(resident_spans[0].tensor_offset == 0 &&
         resident_spans[0].size == 8);
  assert(resident_spans[1].tensor_offset == 8 &&
         resident_spans[1].size == 16);
  assert(!BuildGem5GenericGptqResidentSpans(
      &residency.header, sizeof(residency), 9, 10, 12, page_cache,
      sizeof(page_cache), resident_spans, 2, &resident_span_count));
  parameters.output_features = 1;

  const uint16_t half_scales[] = {UINT16_C(0x3800)};
  auto half_operands = operands;
  half_operands.scales = {half_scales, sizeof(half_scales)};
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
  output[0] = 0.0f;
  assert(RunGem5GenericGptqMatMul(
      parameters, 1, half_operands, &stats));
  assert(output[0] == 4.0f);

  auto truncated = operands;
  truncated.qweight.size = 0;
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  assert(!RunGem5GenericGptqMatMul(parameters, 1, truncated, &stats));
  parameters.quantization_bits = 8;
  assert(!RunGem5GenericGptqMatMul(parameters, 1, operands, &stats));

  auto expert_parameters = Parameters();
  expert_parameters.opcode = 13;
  expert_parameters.input_features = 2;
  expert_parameters.output_features = 2;
  expert_parameters.intermediate_features = 2;
  const float expert_input[] = {1.0f, 2.0f};
  const uint32_t identity_qweight[] = {
      UINT32_C(0x00000001), UINT32_C(0x00000010)};
  const uint32_t sum_qweight[] = {
      UINT32_C(0x00000011), UINT32_C(0x00000011)};
  const uint32_t expert_qzeros[] = {0};
  const float expert_scales[] = {1.0f, 1.0f};
  float gate_output[2] = {};
  float up_output[2] = {};
  float activated[2] = {};
  float expert_output[2] = {};
  const Gem5GenericGptqExpertOperands expert_operands = {
      {expert_input, sizeof(expert_input)},
      IdentityWeights(identity_qweight, expert_qzeros, expert_scales),
      IdentityWeights(sum_qweight, expert_qzeros, expert_scales),
      IdentityWeights(identity_qweight, expert_qzeros, expert_scales),
      {gate_output, sizeof(gate_output)},
      {up_output, sizeof(up_output)},
      {activated, sizeof(activated)},
      {expert_output, sizeof(expert_output)},
  };
  assert(RunGem5GenericGptqExpert(
      expert_parameters, 1, expert_operands, &stats));
  assert(gate_output[0] == 1.0f && gate_output[1] == 2.0f);
  assert(up_output[0] == 3.0f && up_output[1] == 3.0f);
  assert(std::fabs(expert_output[0] -
                   3.0f / (1.0f + std::exp(-1.0f))) < 1e-6f);
  assert(std::fabs(expert_output[1] -
                   6.0f / (1.0f + std::exp(-2.0f))) < 1e-6f);
  assert(stats.operations == 36);
  assert(stats.modeled_cycles > 0);

  const uint32_t doubled_qweight[] = {
      UINT32_C(0x00000002), UINT32_C(0x00000020)};
  float expert_one_gate[2] = {};
  float expert_one_up[2] = {};
  float expert_one_activated[2] = {};
  float expert_one_output[2] = {};
  const Gem5GenericGptqExpertOperands expert_one_operands = {
      {expert_input, sizeof(expert_input)},
      IdentityWeights(doubled_qweight, expert_qzeros, expert_scales),
      IdentityWeights(sum_qweight, expert_qzeros, expert_scales),
      IdentityWeights(doubled_qweight, expert_qzeros, expert_scales),
      {expert_one_gate, sizeof(expert_one_gate)},
      {expert_one_up, sizeof(expert_one_up)},
      {expert_one_activated, sizeof(expert_one_activated)},
      {expert_one_output, sizeof(expert_one_output)},
  };
  assert(RunGem5GenericGptqExpert(
      expert_parameters, 1, expert_one_operands, &stats));
  RoutedWeights routed_weights = {};
  for (size_t projection = 0; projection < 3; ++projection) {
    routed_weights.experts[0][projection] =
        projection == 1
            ? IdentityWeights(sum_qweight, expert_qzeros, expert_scales)
            : IdentityWeights(identity_qweight, expert_qzeros, expert_scales);
    routed_weights.experts[1][projection] =
        projection == 1
            ? IdentityWeights(sum_qweight, expert_qzeros, expert_scales)
            : IdentityWeights(doubled_qweight, expert_qzeros, expert_scales);
  }
  const uint64_t expert_ids[] = {0, 1};
  const float route_weights[] = {0.25f, 0.75f};
  float routed_output[2] = {};
  assert(RunGem5RoutedGptqExperts(
      expert_parameters, 1, {expert_input, sizeof(expert_input)}, expert_ids,
      route_weights, 2, ProvideRoutedWeights, &routed_weights,
      {routed_output, sizeof(routed_output)}, &stats));
  assert(std::fabs(routed_output[0] -
                   (0.25f * expert_output[0] +
                    0.75f * expert_one_output[0])) < 1e-6f);
  assert(std::fabs(routed_output[1] -
                   (0.25f * expert_output[1] +
                    0.75f * expert_one_output[1])) < 1e-6f);
  assert(stats.operations > 72);
  const float two_row_input[] = {1.0f, 2.0f, 1.0f, 2.0f};
  const uint64_t per_row_experts[] = {0, 1};
  const float per_row_weights[] = {1.0f, 1.0f};
  float two_row_output[4] = {};
  assert(RunGem5RoutedGptqExperts(
      expert_parameters, 2, {two_row_input, sizeof(two_row_input)},
      per_row_experts, per_row_weights, 1, ProvideRoutedWeights,
      &routed_weights, {two_row_output, sizeof(two_row_output)}, &stats));
  assert(std::fabs(two_row_output[0] - expert_output[0]) < 1e-6f);
  assert(std::fabs(two_row_output[1] - expert_output[1]) < 1e-6f);
  assert(std::fabs(two_row_output[2] - expert_one_output[0]) < 1e-6f);
  assert(std::fabs(two_row_output[3] - expert_one_output[1]) < 1e-6f);
  expert_parameters.intermediate_features = 0;
  assert(!RunGem5GenericGptqExpert(
      expert_parameters, 1, expert_operands, &stats));
  return 0;
}
