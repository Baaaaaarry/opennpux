#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"
#include "opennpux/model_package.h"
#include "opennpux/npu_weight_ranges.h"

namespace {

struct OwnedWeights {
  std::array<std::vector<uint8_t>, 4> components;
};

Gem5GenericGptqWeights View(const OwnedWeights& weights) {
  return {{weights.components[0].data(), weights.components[0].size()},
          {weights.components[1].data(), weights.components[1].size()},
          {weights.components[2].data(), weights.components[2].size()},
          {weights.components[3].data(), weights.components[3].size()}};
}

Gem5HostGptqWeights HostView(const OwnedWeights& weights) {
  return {{weights.components[0].data(), weights.components[0].size()},
          {weights.components[1].data(), weights.components[1].size()},
          {weights.components[2].data(), weights.components[2].size()},
          {weights.components[3].data(), weights.components[3].size()}};
}

float Float16ToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x3ff;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
    }
  } else if (exponent == 0x1f) {
    bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool ConvertFloatWeight(const std::vector<uint8_t>& bytes, uint32_t data_type,
                        std::vector<float>* output) {
  if (output == nullptr || bytes.empty()) {
    return false;
  }
  size_t element_size = 0;
  if (data_type == OPENNPUX_NPU_DTYPE_FLOAT16 ||
      data_type == OPENNPUX_NPU_DTYPE_BFLOAT16) {
    element_size = sizeof(uint16_t);
  } else if (data_type == OPENNPUX_NPU_DTYPE_INT8) {
    element_size = sizeof(int8_t);
  } else if (data_type == OPENNPUX_NPU_DTYPE_INT32) {
    element_size = sizeof(int32_t);
  } else if (data_type == OPENNPUX_NPU_DTYPE_FLOAT32) {
    element_size = sizeof(float);
  } else {
    return false;
  }
  if (bytes.size() % element_size != 0) {
    return false;
  }
  try {
    output->resize(bytes.size() / element_size);
  } catch (...) {
    return false;
  }
  for (size_t index = 0; index < output->size(); ++index) {
    if (data_type == OPENNPUX_NPU_DTYPE_INT8) {
      (*output)[index] = static_cast<float>(
          static_cast<int8_t>(bytes[index]));
    } else if (data_type == OPENNPUX_NPU_DTYPE_INT32) {
      int32_t value = 0;
      std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
      (*output)[index] = static_cast<float>(value);
    } else if (data_type == OPENNPUX_NPU_DTYPE_FLOAT32) {
      std::memcpy(&(*output)[index], bytes.data() + index * sizeof(float),
                  sizeof(float));
    } else {
      uint16_t value = 0;
      std::memcpy(&value, bytes.data() + index * sizeof(value), sizeof(value));
      if (data_type == OPENNPUX_NPU_DTYPE_BFLOAT16) {
        const uint32_t bits = static_cast<uint32_t>(value) << 16;
        std::memcpy(&(*output)[index], &bits, sizeof(bits));
      } else {
        (*output)[index] = Float16ToFloat(value);
      }
    }
    if (!std::isfinite((*output)[index])) {
      return false;
    }
  }
  return true;
}

}  // namespace

struct Gem5HostWeightProvider::Impl {
  bool LoadGroup(uint32_t command_id, uint32_t role_id, uint64_t expert_id,
                 uint32_t slot_id, OwnedWeights* owned,
                 Gem5GenericGptqWeights* view) {
    if (!loaded || owned == nullptr || view == nullptr) {
      return false;
    }
    opennpux_npu_gptq_weight_ranges group = {};
    if (opennpux_npu_weight_ranges_find_gptq(
            &ranges, command_id, role_id, expert_id, slot_id, &group) != 0) {
      return false;
    }
    const opennpux_npu_weight_range_record* records[] = {
        group.qweight, group.qzeros, group.scales, group.g_idx};
    for (size_t index = 0; index < 4; ++index) {
      auto& bytes = owned->components[index];
      bytes.clear();
      if (records[index] == nullptr) {
        continue;
      }
      if (records[index]->byte_size == 0 ||
          records[index]->byte_size > max_component_size ||
          records[index]->byte_size > SIZE_MAX) {
        return false;
      }
      try {
        bytes.resize(static_cast<size_t>(records[index]->byte_size));
      } catch (...) {
        return false;
      }
      if (opennpux_model_package_read_shard_range(
              manifest_path.c_str(), &model, records[index]->shard_index,
              records[index]->file_offset, bytes.data(), bytes.size()) != 0) {
        return false;
      }
    }
    *view = View(*owned);
    return true;
  }

  std::string manifest_path;
  opennpux_model_package_info model = {};
  opennpux_npu_weight_ranges ranges = {};
  std::array<OwnedWeights, 3> projections;
  OwnedWeights gate;
  OwnedWeights up;
  OwnedWeights down;
  size_t max_component_size = 0;
  uint32_t routed_command_id = UINT32_MAX;
  std::string last_error;
  bool loaded = false;
};

Gem5HostWeightProvider::Gem5HostWeightProvider()
    : impl_(std::make_unique<Impl>()) {}

Gem5HostWeightProvider::~Gem5HostWeightProvider() { Reset(); }

bool Gem5HostWeightProvider::Load(const std::string& manifest_path,
                                  const std::string& weight_ranges_path,
                                  size_t max_component_size) {
  Reset();
  if (manifest_path.empty() || weight_ranges_path.empty() ||
      max_component_size == 0 ||
      opennpux_model_package_load(manifest_path.c_str(), &impl_->model) != 0 ||
      opennpux_npu_weight_ranges_load(weight_ranges_path.c_str(),
                                      &impl_->ranges) != 0 ||
      impl_->ranges.header->shard_count != impl_->model.shard_count) {
    Reset();
    return false;
  }
  impl_->manifest_path = manifest_path;
  impl_->max_component_size = max_component_size;
  impl_->loaded = true;
  return true;
}

void Gem5HostWeightProvider::Reset() {
  if (impl_ == nullptr) {
    return;
  }
  opennpux_npu_weight_ranges_unload(&impl_->ranges);
  impl_->manifest_path.clear();
  impl_->model = {};
  impl_->projections = {};
  impl_->gate = {};
  impl_->up = {};
  impl_->down = {};
  impl_->max_component_size = 0;
  impl_->routed_command_id = UINT32_MAX;
  impl_->last_error.clear();
  impl_->loaded = false;
}

bool Gem5HostWeightProvider::LoadProjection(
    uint32_t command_id, uint32_t role_id, uint64_t expert_id,
    uint32_t slot_id, Gem5HostGptqWeights* weights, uint32_t bank) {
  if (weights == nullptr || bank >= impl_->projections.size()) {
    return false;
  }
  Gem5GenericGptqWeights generic = {};
  if (!impl_->LoadGroup(command_id, role_id, expert_id, slot_id,
                        &impl_->projections[bank], &generic)) {
    return false;
  }
  *weights = HostView(impl_->projections[bank]);
  return true;
}

bool Gem5HostWeightProvider::LoadFloatWeight(
    uint32_t command_id, uint32_t role_id, uint64_t expert_id,
    uint32_t slot_id, std::vector<float>* weights) {
  if (!impl_->loaded || weights == nullptr) {
    impl_->last_error = "float-weight provider is not loaded";
    return false;
  }
  impl_->last_error.clear();
  const opennpux_npu_weight_range_record* record = nullptr;
  if (opennpux_npu_weight_range_find_slot(
          &impl_->ranges, command_id, role_id,
          OPENNPUX_NPU_WEIGHT_COMPONENT_WEIGHT, expert_id, slot_id,
          &record) != 0) {
    impl_->last_error = "float-weight range lookup failed errno=" +
                        std::to_string(errno);
    return false;
  }
  if (record->byte_size == 0 ||
      record->byte_size > impl_->max_component_size ||
      record->byte_size > SIZE_MAX) {
    impl_->last_error = "invalid float-weight range size=" +
                        std::to_string(record->byte_size);
    return false;
  }
  std::vector<uint8_t> bytes;
  try {
    bytes.resize(static_cast<size_t>(record->byte_size));
  } catch (...) {
    impl_->last_error = "float-weight allocation failed bytes=" +
                        std::to_string(record->byte_size);
    return false;
  }
  if (opennpux_model_package_read_shard_range(
          impl_->manifest_path.c_str(), &impl_->model, record->shard_index,
          record->file_offset, bytes.data(), bytes.size()) != 0) {
    impl_->last_error = "float-weight shard read failed shard=" +
                        std::to_string(record->shard_index) + " offset=" +
                        std::to_string(record->file_offset) + " bytes=" +
                        std::to_string(record->byte_size) + " errno=" +
                        std::to_string(errno);
    return false;
  }
  const uint32_t data_type = static_cast<uint32_t>(
      (record->flags & OPENNPUX_NPU_WEIGHT_DTYPE_MASK) >>
      OPENNPUX_NPU_WEIGHT_DTYPE_SHIFT);
  if (!ConvertFloatWeight(bytes, data_type, weights)) {
    impl_->last_error = "float-weight conversion failed dtype=" +
                        std::to_string(data_type) + " bytes=" +
                        std::to_string(record->byte_size);
    return false;
  }
  return true;
}

bool Gem5HostWeightProvider::FindGptqBindings(
    uint32_t command_id,
    std::vector<Gem5HostWeightBinding>* bindings) const {
  if (!impl_->loaded || bindings == nullptr) {
    return false;
  }
  bindings->clear();
  const opennpux_npu_weight_range_record* records = nullptr;
  uint32_t record_count = 0;
  if (opennpux_npu_weight_ranges_for_command(
          &impl_->ranges, command_id, &records, &record_count) != 0) {
    return false;
  }
  std::vector<std::tuple<uint32_t, uint64_t, uint32_t>> seen;
  for (uint32_t index = 0; index < record_count; ++index) {
    const auto& record = records[index];
    if (record.component_id != OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT) {
      continue;
    }
    const uint32_t slot = static_cast<uint32_t>(
        record.flags & OPENNPUX_NPU_WEIGHT_SLOT_MASK);
    const auto key = std::make_tuple(record.role_id, record.expert_id, slot);
    if (std::find(seen.begin(), seen.end(), key) != seen.end()) {
      continue;
    }
    opennpux_npu_gptq_weight_ranges group = {};
    if (opennpux_npu_weight_ranges_find_gptq(
            &impl_->ranges, command_id, record.role_id, record.expert_id,
            slot, &group) != 0) {
      continue;
    }
    seen.push_back(key);
    bindings->push_back({record.role_id, record.expert_id, slot});
  }
  return !bindings->empty();
}

bool Gem5HostWeightProvider::FindFloatBindings(
    uint32_t command_id,
    std::vector<Gem5HostWeightBinding>* bindings) const {
  if (!impl_->loaded || bindings == nullptr) {
    return false;
  }
  bindings->clear();
  const opennpux_npu_weight_range_record* records = nullptr;
  uint32_t record_count = 0;
  if (opennpux_npu_weight_ranges_for_command(
          &impl_->ranges, command_id, &records, &record_count) != 0) {
    return false;
  }
  for (uint32_t index = 0; index < record_count; ++index) {
    const auto& record = records[index];
    if (record.component_id != OPENNPUX_NPU_WEIGHT_COMPONENT_WEIGHT) {
      continue;
    }
    const Gem5HostWeightBinding binding = {
        record.role_id, record.expert_id,
        static_cast<uint32_t>(record.flags & OPENNPUX_NPU_WEIGHT_SLOT_MASK)};
    const auto duplicate = std::find_if(
        bindings->begin(), bindings->end(), [&](const auto& existing) {
          return existing.role_id == binding.role_id &&
                 existing.expert_id == binding.expert_id &&
                 existing.slot_id == binding.slot_id;
        });
    if (duplicate == bindings->end()) {
      bindings->push_back(binding);
    }
  }
  return !bindings->empty();
}

bool Gem5HostWeightProvider::ConfigureRoutedExpert(uint32_t command_id) {
  if (!impl_->loaded || command_id >= impl_->ranges.header->command_count) {
    return false;
  }
  impl_->routed_command_id = command_id;
  return true;
}

bool Gem5HostWeightProvider::ProvideRoutedExpert(
    void* opaque, uint64_t expert_id, Gem5GenericGptqWeights* gate,
    Gem5GenericGptqWeights* up, Gem5GenericGptqWeights* down) {
  auto* provider = static_cast<Gem5HostWeightProvider*>(opaque);
  if (provider == nullptr || !provider->impl_->loaded ||
      provider->impl_->routed_command_id == UINT32_MAX) {
    return false;
  }
  return provider->impl_->LoadGroup(
             provider->impl_->routed_command_id,
             OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT, expert_id,
             OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ, &provider->impl_->gate,
             gate) &&
         provider->impl_->LoadGroup(
             provider->impl_->routed_command_id,
             OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT, expert_id,
             OPENNPUX_NPU_WEIGHT_SLOT_UP_PROJ, &provider->impl_->up, up) &&
         provider->impl_->LoadGroup(
             provider->impl_->routed_command_id,
             OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT, expert_id,
             OPENNPUX_NPU_WEIGHT_SLOT_DOWN_PROJ, &provider->impl_->down,
             down);
}

bool Gem5HostWeightProvider::loaded() const {
  return impl_ != nullptr && impl_->loaded;
}

const std::string& Gem5HostWeightProvider::last_error() const {
  return impl_->last_error;
}
