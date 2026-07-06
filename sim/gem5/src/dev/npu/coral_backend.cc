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

void
CoralBackend::readLocalExtmem(Addr addr, void *data, size_t size)
{
    panic("Coral backend does not implement local EXTMEM reads");
}

void
CoralBackend::writeLocalExtmem(Addr addr, const void *data, size_t size)
{
    panic("Coral backend does not implement local EXTMEM writes");
}

} // namespace gem5
