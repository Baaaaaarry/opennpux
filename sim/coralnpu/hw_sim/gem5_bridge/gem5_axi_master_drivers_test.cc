#include "hw_sim/gem5_bridge/gem5_axi_master_drivers.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

double
sc_time_stamp()
{
  return 0;
}

namespace {

void
Check(bool condition, const char* message)
{
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

struct ReadyDroppingModel {
  uint8_t* clock;
  uint8_t* response_valid;
  uint8_t* response_ready;

  void eval() {
    if (*clock && *response_valid && *response_ready) {
      *response_ready = 0;
    }
  }
};

void
TestDeferredRead()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0x20000000;
  uint8_t prot = 0;
  uint8_t id = 3;
  uint8_t len = 0;
  uint8_t size = 2;
  uint8_t burst = 1;
  uint8_t lock = 0;
  uint8_t cache = 0;
  uint8_t qos = 0;
  uint8_t region = 0;
  uint8_t addr_ready = 0;
  uint8_t data_valid = 0;
  VlWide<4> data = {};
  uint8_t data_id = 0;
  uint8_t data_resp = 0;
  uint8_t data_last = 0;
  uint8_t data_ready = 1;
  ReadyDroppingModel model{
      &clock_signal, &data_valid, &data_ready};
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterReadDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &data_id, &data_resp, &data_last, &data_ready);

  int requests = 0;
  driver.RegisterDeferredCallback([&](const AxiAddr& request) {
    ++requests;
    Check(request.addr_bits_addr == addr, "read address changed");
    Check(request.addr_bits_id == id, "read ID changed");
  });

  addr_valid = 1;
  clock.Step();
  Check(requests == 1, "read request was not captured");
  Check(driver.HasDeferredRequest(), "read request is not pending");

  clock.Step();
  Check(requests == 1, "held ARVALID replayed a read request");

  AxiRData response = {};
  response.read_data_bits_data[0] = 0x12345678;
  response.read_data_bits_id = id;
  response.read_data_bits_resp = 0;
  response.read_data_bits_last = 1;
  driver.QueueResponse(response);

  clock.Step();
  Check(data_valid == 1, "read response was not presented");
  Check(data[0] == 0x12345678, "read response data changed");

  clock.Step();
  Check(data_ready == 0, "read READY did not drop after rising edge");
  Check(data_valid == 0, "accepted read response was not retired");
  Check(!driver.HasDeferredRequest(), "read request remained pending");

  addr_valid = 0;
  clock.Step();
  data_ready = 1;
  addr = 0x20000004;
  addr_valid = 1;
  clock.Step();
  Check(requests == 2, "next read request was not accepted");
}

void
TestIndependentWriteChannels()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0x20000000;
  uint8_t prot = 0;
  uint8_t id = 5;
  uint8_t len = 0;
  uint8_t size = 2;
  uint8_t burst = 1;
  uint8_t lock = 0;
  uint8_t cache = 0;
  uint8_t qos = 0;
  uint8_t region = 0;
  uint8_t addr_ready = 0;
  uint8_t data_valid = 0;
  VlWide<4> data = {};
  uint16_t strb = 0x000f;
  uint8_t last = 1;
  uint8_t data_ready = 0;
  uint8_t resp_valid = 0;
  uint8_t resp_id = 0;
  uint8_t resp = 0;
  uint8_t resp_ready = 1;
  ReadyDroppingModel model{
      &clock_signal, &resp_valid, &resp_ready};
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterWriteDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &strb, &last, &data_ready, &resp_valid, &resp_id, &resp,
      &resp_ready);

  int requests = 0;
  uint32_t captured_addr = 0;
  uint32_t captured_data = 0;
  driver.RegisterDeferredCallback(
      [&](const AxiAddr& request_addr, const AxiWData& request_data) {
        ++requests;
        captured_addr = request_addr.addr_bits_addr;
        captured_data = request_data.write_data_bits_data[0];
      });

  clock.Step();
  Check(addr_ready == 1 && data_ready == 1,
        "write channels did not become ready");

  addr_valid = 1;
  clock.Step();
  Check(requests == 0, "write submitted before W channel arrived");
  Check(addr_ready == 0 && data_ready == 1,
        "AW and W readiness was not independent");

  addr_valid = 0;
  data[0] = 42;
  data_valid = 1;
  clock.Step();
  Check(requests == 1, "write was not submitted after AW and W");
  Check(captured_addr == 0x20000000, "captured write address changed");
  Check(captured_data == 42, "captured write data changed");

  clock.Step();
  Check(requests == 1, "held WVALID replayed a write request");

  AxiWResp response = {};
  response.write_resp_bits_id = id;
  response.write_resp_bits_resp = 0;
  driver.QueueResponse(response);
  clock.Step();
  Check(resp_valid == 1, "write response was not presented");

  clock.Step();
  Check(resp_ready == 0, "write READY did not drop after rising edge");
  Check(resp_valid == 0, "accepted write response was not retired");
  Check(!driver.HasDeferredRequest(), "write request remained pending");

  data_valid = 0;
  resp_ready = 1;
  data[0] = 0x4e505544;
  data_valid = 1;
  clock.Step();
  Check(requests == 1, "write submitted before second AW channel");

  data_valid = 0;
  addr = 0x20000008;
  addr_valid = 1;
  clock.Step();
  Check(requests == 2, "W-before-AW write was not submitted");
  Check(captured_addr == 0x20000008, "second write address changed");
  Check(captured_data == 0x4e505544, "second write data changed");
}

}  // namespace

int
main()
{
  TestDeferredRead();
  TestIndependentWriteChannels();
  std::puts("PASS: gem5 Coral AXI master adapter");
  return 0;
}
