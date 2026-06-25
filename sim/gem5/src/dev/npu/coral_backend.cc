#include "dev/npu/coral_backend.hh"

#include "base/logging.hh"

namespace gem5
{

const CoralDmaRequest &
CoralBackend::dmaRequest() const
{
    panic("Coral backend does not implement DMA requests");
}

void
CoralBackend::completeDma(const uint8_t *data, size_t size, bool error)
{
    panic("Coral backend does not implement DMA completion");
}

} // namespace gem5
