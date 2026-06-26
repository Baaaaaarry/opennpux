#include "hw_sim/gem5_bridge/gem5_axi_master_drivers.h"
#include "hw_sim/gem5_bridge/gem5_dma_request_builder.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

struct PassiveModel {
  void eval() {}
};

struct RandomReadyModel {
  uint8_t* clock;
  uint8_t* response_valid;
  uint8_t* response_ready;
  uint32_t state;
  uint8_t previous_clock = 0;
  int handshakes = 0;

  uint32_t Next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  }

  void eval() {
    if (!previous_clock && *clock) {
      if (*response_valid && *response_ready) {
        ++handshakes;
        *response_ready = 0;
      }
    } else if (previous_clock && !*clock) {
      *response_ready = (Next() & 3u) != 0;
    }
    previous_clock = *clock;
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
      [&](const AxiAddr& request_addr,
          const std::vector<AxiWData>& request_data) {
        ++requests;
        Check(request_data.size() == 1,
              "single-beat write captured wrong beat count");
        captured_addr = request_addr.addr_bits_addr;
        captured_data = request_data[0].write_data_bits_data[0];
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

void
TestDmaRequestValidation()
{
  AxiAddr addr = {};
  addr.addr_bits_addr = 0x20000004;
  addr.addr_bits_id = 7;
  addr.addr_bits_size = 2;
  addr.addr_bits_len = 0;
  addr.addr_bits_burst = 1;
  coral_gem5_dma_request request = {};
  uint32_t beat_size = 0;
  uint32_t beat_count = 0;

  Check(BuildGem5DmaReadRequest(
            addr, &request, &beat_size, &beat_count),
        "valid read request was rejected");
  Check(request.size == 4 && beat_size == 4 && beat_count == 1,
        "valid read request was encoded incorrectly");

  addr.addr_bits_len = 3;
  Check(BuildGem5DmaReadRequest(
            addr, &request, &beat_size, &beat_count),
        "valid read burst was rejected");
  Check(request.size == 16 && beat_size == 4 && beat_count == 4,
        "read burst was encoded incorrectly");

  addr.addr_bits_burst = 0;
  Check(!BuildGem5DmaReadRequest(
            addr, &request, &beat_size, &beat_count),
        "non-INCR read burst was accepted");
  addr.addr_bits_burst = 1;
  addr.addr_bits_addr = 0x20000ffcu;
  addr.addr_bits_len = 1;
  Check(!BuildGem5DmaReadRequest(
            addr, &request, &beat_size, &beat_count),
        "read burst crossing 4 KiB was accepted");
  addr.addr_bits_addr = 0x2000000f;
  addr.addr_bits_len = 0;
  Check(!BuildGem5DmaReadRequest(
            addr, &request, &beat_size, &beat_count),
        "read beat crossing the AXI data word was accepted");

  addr.addr_bits_addr = 0x20000004;
  addr.addr_bits_len = 0;
  AxiWData data = {};
  data.write_data_bits_data[1] = 0x11223344;
  data.write_data_bits_strb = 0x00f0;
  data.write_data_bits_last = 1;
  Check(BuildGem5DmaWriteRequest(addr, std::vector<AxiWData>{data}, &request),
        "valid write request was rejected");
  Check(request.size == 4 && request.data[0] == 0x44 &&
            request.data[3] == 0x11,
        "valid write request was encoded incorrectly");

  data.write_data_bits_strb = 0x0070;
  Check(!BuildGem5DmaWriteRequest(addr, std::vector<AxiWData>{data},
                                  &request),
        "partial write strobe was accepted without byte enables");
  data.write_data_bits_strb = 0x00f0;
  data.write_data_bits_last = 0;
  Check(!BuildGem5DmaWriteRequest(addr, std::vector<AxiWData>{data},
                                  &request),
        "non-final write beat was accepted");

  addr.addr_bits_addr = 0x20000000;
  addr.addr_bits_len = 3;
  std::vector<AxiWData> burst(4);
  for (uint32_t beat = 0; beat < burst.size(); ++beat) {
    burst[beat].write_data_bits_data[beat] = 0x44332211u + beat;
    burst[beat].write_data_bits_strb = 0x000fu << (beat * 4);
    burst[beat].write_data_bits_last = beat + 1 == burst.size();
  }
  Check(BuildGem5DmaWriteRequest(addr, burst, &request),
        "valid write burst was rejected");
  Check(request.size == 16 && request.data[0] == 0x11 &&
            request.data[4] == 0x12 && request.data[8] == 0x13 &&
            request.data[12] == 0x14,
        "write burst payload was encoded incorrectly");

  burst[2].write_data_bits_last = 1;
  Check(!BuildGem5DmaWriteRequest(addr, burst, &request),
        "early WLAST write burst was accepted");
  burst[2].write_data_bits_last = 0;
  burst.pop_back();
  Check(!BuildGem5DmaWriteRequest(addr, burst, &request),
        "short write burst was accepted");

  addr.addr_bits_addr = 0x20000ffcu;
  addr.addr_bits_len = 1;
  burst.resize(2);
  burst[0].write_data_bits_data[3] = 0xaa55aa55u;
  burst[0].write_data_bits_strb = 0xf000u;
  burst[0].write_data_bits_last = 0;
  burst[1].write_data_bits_data[0] = 0x55aa55aau;
  burst[1].write_data_bits_strb = 0x000fu;
  burst[1].write_data_bits_last = 1;
  Check(!BuildGem5DmaWriteRequest(addr, burst, &request),
        "write burst crossing 4 KiB was accepted");
}

void
TestWriteBurstCollection()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0x20000000;
  uint8_t prot = 0;
  uint8_t id = 11;
  uint8_t len = 3;
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
  uint8_t last = 0;
  uint8_t data_ready = 0;
  uint8_t resp_valid = 0;
  uint8_t resp_id = 0;
  uint8_t resp = 0;
  uint8_t resp_ready = 1;
  PassiveModel model;
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterWriteDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &strb, &last, &data_ready, &resp_valid, &resp_id, &resp,
      &resp_ready);

  int requests = 0;
  std::vector<uint32_t> captured_data;
  driver.RegisterDeferredCallback(
      [&](const AxiAddr& request_addr,
          const std::vector<AxiWData>& request_data) {
        ++requests;
        Check(request_addr.addr_bits_len == 3,
              "write burst address length changed");
        Check(request_data.size() == 4,
              "write burst captured wrong beat count");
        captured_data.clear();
        for (const AxiWData& beat : request_data) {
          captured_data.push_back(beat.write_data_bits_data[0]);
        }
      });

  clock.Step();
  data_valid = 1;
  for (uint32_t beat = 0; beat < 4; ++beat) {
    data[0] = 0xcafe0000u | beat;
    last = beat == 3;
    clock.Step();
    Check(requests == 0, "write burst submitted before AW arrived");
  }
  data_valid = 0;
  Check(data_ready == 0, "write burst accepted data after WLAST");

  addr_valid = 1;
  clock.Step();
  addr_valid = 0;
  Check(requests == 1, "write burst was not submitted after AW");
  Check(captured_data.size() == 4 && captured_data[0] == 0xcafe0000u &&
            captured_data[3] == 0xcafe0003u,
        "write burst data changed");

  AxiWResp response = {};
  response.write_resp_bits_id = id;
  driver.QueueResponse(response);
  int timeout = 16;
  while (driver.HasDeferredRequest() && --timeout > 0) {
    clock.Step();
  }
  Check(timeout > 0, "write burst response timed out");
}

void
TestReadBurstResponseRetirement()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0x20000000;
  uint8_t prot = 0;
  uint8_t id = 9;
  uint8_t len = 3;
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
  PassiveModel model;
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterReadDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &data_id, &data_resp, &data_last, &data_ready);

  int requests = 0;
  driver.RegisterDeferredCallback(
      [&](const AxiAddr&) { ++requests; });
  addr_valid = 1;
  clock.Step();
  addr_valid = 0;
  Check(requests == 1 && driver.HasDeferredRequest(),
        "read burst request was not captured");

  for (uint32_t beat = 0; beat < 4; ++beat) {
    AxiRData response = {};
    response.read_data_bits_data[beat] = 0x100 + beat;
    response.read_data_bits_id = id;
    response.read_data_bits_last = beat == 3;
    driver.QueueResponse(response);
  }

  clock.Step();
  Check(data_valid == 1, "first read burst beat was not presented");
  for (uint32_t beat = 0; beat < 4; ++beat) {
    clock.Step();
    Check(driver.HasDeferredRequest() == (beat != 3),
          "read burst retired before RLAST or remained after RLAST");
  }
}

void
TestErrorResponseRetirement()
{
  VerilatedContext context;
  uint8_t read_clock_signal = 0;
  uint8_t read_addr_valid = 0;
  uint32_t read_addr = 0x20000000;
  uint8_t read_prot = 0;
  uint8_t read_id = 4;
  uint8_t read_len = 0;
  uint8_t read_size = 2;
  uint8_t read_burst = 1;
  uint8_t read_lock = 0;
  uint8_t read_cache = 0;
  uint8_t read_qos = 0;
  uint8_t read_region = 0;
  uint8_t read_addr_ready = 0;
  uint8_t read_data_valid = 0;
  VlWide<4> read_data = {};
  uint8_t read_data_id = 0;
  uint8_t read_data_resp = 0;
  uint8_t read_data_last = 0;
  uint8_t read_data_ready = 1;
  PassiveModel read_model;
  Clock read_clock(&context, &read_clock_signal, &read_model);
  Gem5AxiMasterReadDriver read_driver(
      &read_clock, &read_addr_valid, &read_addr, &read_prot, &read_id,
      &read_len, &read_size, &read_burst, &read_lock, &read_cache,
      &read_qos, &read_region, &read_addr_ready, &read_data_valid,
      &read_data, &read_data_id, &read_data_resp, &read_data_last,
      &read_data_ready);

  int read_requests = 0;
  read_driver.RegisterDeferredCallback(
      [&](const AxiAddr&) { ++read_requests; });
  read_addr_valid = 1;
  read_clock.Step();
  read_addr_valid = 0;
  Check(read_requests == 1 && read_driver.HasDeferredRequest(),
        "read error request was not captured");

  AxiRData read_error = {};
  read_error.read_data_bits_id = read_id;
  read_error.read_data_bits_resp = 2;
  read_error.read_data_bits_last = 1;
  read_driver.QueueResponse(read_error);
  read_clock.Step();
  Check(read_data_valid == 1 && read_data_resp == 2 &&
            read_data_last == 1,
        "read error response was not presented");
  read_clock.Step();
  Check(!read_driver.HasDeferredRequest(),
        "read error response did not release pending request");

  uint8_t write_clock_signal = 0;
  uint8_t write_addr_valid = 0;
  uint32_t write_addr = 0x20000000;
  uint8_t write_prot = 0;
  uint8_t write_id = 6;
  uint8_t write_len = 0;
  uint8_t write_size = 2;
  uint8_t write_burst = 1;
  uint8_t write_lock = 0;
  uint8_t write_cache = 0;
  uint8_t write_qos = 0;
  uint8_t write_region = 0;
  uint8_t write_addr_ready = 0;
  uint8_t write_data_valid = 0;
  VlWide<4> write_data = {};
  uint16_t write_strb = 0x000f;
  uint8_t write_last = 1;
  uint8_t write_data_ready = 0;
  uint8_t write_resp_valid = 0;
  uint8_t write_resp_id = 0;
  uint8_t write_resp = 0;
  uint8_t write_resp_ready = 1;
  PassiveModel write_model;
  Clock write_clock(&context, &write_clock_signal, &write_model);
  Gem5AxiMasterWriteDriver write_driver(
      &write_clock, &write_addr_valid, &write_addr, &write_prot,
      &write_id, &write_len, &write_size, &write_burst, &write_lock,
      &write_cache, &write_qos, &write_region, &write_addr_ready,
      &write_data_valid, &write_data, &write_strb, &write_last,
      &write_data_ready, &write_resp_valid, &write_resp_id,
      &write_resp, &write_resp_ready);

  int write_requests = 0;
  write_driver.RegisterDeferredCallback(
      [&](const AxiAddr&, const std::vector<AxiWData>&) {
        ++write_requests;
      });
  write_clock.Step();
  write_addr_valid = 1;
  write_data_valid = 1;
  write_data[0] = 0x12345678;
  write_clock.Step();
  write_addr_valid = 0;
  write_data_valid = 0;
  Check(write_requests == 1 && write_driver.HasDeferredRequest(),
        "write error request was not captured");

  AxiWResp write_error = {};
  write_error.write_resp_bits_id = write_id;
  write_error.write_resp_bits_resp = 2;
  write_driver.QueueResponse(write_error);
  write_clock.Step();
  Check(write_resp_valid == 1 && write_resp == 2 &&
            write_resp_id == write_id,
        "write error response was not presented");
  write_clock.Step();
  Check(!write_driver.HasDeferredRequest(),
        "write error response did not release pending request");
}

void
TestSingleOutstandingBackpressure()
{
  VerilatedContext context;
  uint8_t read_clock_signal = 0;
  uint8_t read_addr_valid = 0;
  uint32_t read_addr = 0x20000000;
  uint8_t read_prot = 0;
  uint8_t read_id = 1;
  uint8_t read_len = 0;
  uint8_t read_size = 2;
  uint8_t read_burst = 1;
  uint8_t read_lock = 0;
  uint8_t read_cache = 0;
  uint8_t read_qos = 0;
  uint8_t read_region = 0;
  uint8_t read_addr_ready = 0;
  uint8_t read_data_valid = 0;
  VlWide<4> read_data = {};
  uint8_t read_data_id = 0;
  uint8_t read_data_resp = 0;
  uint8_t read_data_last = 0;
  uint8_t read_data_ready = 1;
  PassiveModel read_model;
  Clock read_clock(&context, &read_clock_signal, &read_model);
  Gem5AxiMasterReadDriver read_driver(
      &read_clock, &read_addr_valid, &read_addr, &read_prot, &read_id,
      &read_len, &read_size, &read_burst, &read_lock, &read_cache,
      &read_qos, &read_region, &read_addr_ready, &read_data_valid,
      &read_data, &read_data_id, &read_data_resp, &read_data_last,
      &read_data_ready);

  int read_requests = 0;
  read_driver.RegisterDeferredCallback(
      [&](const AxiAddr&) { ++read_requests; });
  read_addr_valid = 1;
  read_clock.Step();
  read_addr_valid = 0;
  read_clock.Step();
  Check(read_requests == 1 && read_addr_ready == 0,
        "read driver did not backpressure a pending request");

  read_addr = 0x20000004;
  read_addr_valid = 1;
  read_clock.Step();
  Check(read_requests == 1,
        "read driver accepted a second outstanding request");
  read_addr_valid = 0;

  AxiRData read_response = {};
  read_response.read_data_bits_id = read_id;
  read_response.read_data_bits_last = 1;
  read_driver.QueueResponse(read_response);
  while (read_driver.HasDeferredRequest()) {
    read_clock.Step();
  }
  read_clock.Step();
  Check(read_addr_ready == 1,
        "read driver did not re-enable AR after response");

  uint8_t write_clock_signal = 0;
  uint8_t write_addr_valid = 0;
  uint32_t write_addr = 0x20000000;
  uint8_t write_prot = 0;
  uint8_t write_id = 2;
  uint8_t write_len = 0;
  uint8_t write_size = 2;
  uint8_t write_burst = 1;
  uint8_t write_lock = 0;
  uint8_t write_cache = 0;
  uint8_t write_qos = 0;
  uint8_t write_region = 0;
  uint8_t write_addr_ready = 0;
  uint8_t write_data_valid = 0;
  VlWide<4> write_data = {};
  uint16_t write_strb = 0x000f;
  uint8_t write_last = 1;
  uint8_t write_data_ready = 0;
  uint8_t write_resp_valid = 0;
  uint8_t write_resp_id = 0;
  uint8_t write_resp = 0;
  uint8_t write_resp_ready = 1;
  PassiveModel write_model;
  Clock write_clock(&context, &write_clock_signal, &write_model);
  Gem5AxiMasterWriteDriver write_driver(
      &write_clock, &write_addr_valid, &write_addr, &write_prot,
      &write_id, &write_len, &write_size, &write_burst, &write_lock,
      &write_cache, &write_qos, &write_region, &write_addr_ready,
      &write_data_valid, &write_data, &write_strb, &write_last,
      &write_data_ready, &write_resp_valid, &write_resp_id,
      &write_resp, &write_resp_ready);

  int write_requests = 0;
  write_driver.RegisterDeferredCallback(
      [&](const AxiAddr&, const std::vector<AxiWData>&) {
        ++write_requests;
      });
  write_clock.Step();
  write_addr_valid = 1;
  write_data_valid = 1;
  write_data[0] = 0x1;
  write_clock.Step();
  write_addr_valid = 0;
  write_data_valid = 0;
  write_clock.Step();
  Check(write_requests == 1 && write_addr_ready == 0 &&
            write_data_ready == 0,
        "write driver did not backpressure a pending request");

  write_addr = 0x20000004;
  write_data[0] = 0x2;
  write_addr_valid = 1;
  write_data_valid = 1;
  write_clock.Step();
  Check(write_requests == 1,
        "write driver accepted a second outstanding request");
  write_addr_valid = 0;
  write_data_valid = 0;

  AxiWResp write_response = {};
  write_response.write_resp_bits_id = write_id;
  write_driver.QueueResponse(write_response);
  while (write_driver.HasDeferredRequest()) {
    write_clock.Step();
  }
  write_clock.Step();
  Check(write_addr_ready == 1 && write_data_ready == 1,
        "write driver did not re-enable AW/W after response");
}

void
TestRandomizedReadTiming()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0;
  uint8_t prot = 0;
  uint8_t id = 0;
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
  uint8_t data_ready = 0;
  RandomReadyModel model{
      &clock_signal, &data_valid, &data_ready, 0x1badb002};
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterReadDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &data_id, &data_resp, &data_last, &data_ready);

  int requests = 0;
  uint32_t captured_addr = 0;
  uint8_t captured_id = 0;
  driver.RegisterDeferredCallback([&](const AxiAddr& request) {
    ++requests;
    captured_addr = request.addr_bits_addr;
    captured_id = request.addr_bits_id;
  });

  for (int transaction = 0; transaction < 64; ++transaction) {
    addr = 0x20000000 + transaction * 4;
    id = transaction & 0xf;
    addr_valid = 1;
    clock.Step();
    Check(requests == transaction + 1,
          "random read request was not captured exactly once");
    Check(captured_addr == addr && captured_id == id,
          "random read request fields changed");

    const int held_cycles = model.Next() & 3u;
    for (int cycle = 0; cycle < held_cycles; ++cycle) {
      clock.Step();
      Check(requests == transaction + 1,
            "random held ARVALID replayed a request");
    }
    addr_valid = 0;
    clock.Step();

    const int completion_delay = model.Next() & 7u;
    for (int cycle = 0; cycle < completion_delay; ++cycle) {
      clock.Step();
    }

    AxiRData response = {};
    response.read_data_bits_data[0] = 0xa5000000u | transaction;
    response.read_data_bits_id = id;
    response.read_data_bits_last = 1;
    driver.QueueResponse(response);

    int timeout = 256;
    while (driver.HasDeferredRequest() && --timeout > 0) {
      clock.Step();
    }
    Check(timeout > 0, "random read response timed out");
    Check(data_valid == 0, "random read response was not retired");
  }

  Check(model.handshakes == 64, "random read handshake count changed");
}

void
TestRandomizedWriteTiming()
{
  VerilatedContext context;
  uint8_t clock_signal = 0;
  uint8_t addr_valid = 0;
  uint32_t addr = 0;
  uint8_t prot = 0;
  uint8_t id = 0;
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
  uint8_t resp_ready = 0;
  RandomReadyModel model{
      &clock_signal, &resp_valid, &resp_ready, 0xc001d00d};
  Clock clock(&context, &clock_signal, &model);
  Gem5AxiMasterWriteDriver driver(
      &clock, &addr_valid, &addr, &prot, &id, &len, &size, &burst,
      &lock, &cache, &qos, &region, &addr_ready, &data_valid, &data,
      &strb, &last, &data_ready, &resp_valid, &resp_id, &resp,
      &resp_ready);

  int requests = 0;
  uint32_t captured_addr = 0;
  uint32_t captured_data = 0;
  uint8_t captured_id = 0;
  driver.RegisterDeferredCallback(
      [&](const AxiAddr& request_addr,
          const std::vector<AxiWData>& request_data) {
        ++requests;
        Check(request_data.size() == 1,
              "random write captured wrong beat count");
        captured_addr = request_addr.addr_bits_addr;
        captured_data = request_data[0].write_data_bits_data[0];
        captured_id = request_addr.addr_bits_id;
      });

  clock.Step();
  for (int transaction = 0; transaction < 64; ++transaction) {
    addr = 0x20000000 + transaction * 4;
    id = transaction & 0xf;
    data[0] = 0x5a000000u | transaction;

    const bool address_first = (model.Next() & 1u) != 0;
    if (address_first) {
      addr_valid = 1;
      clock.Step();
      addr_valid = 0;
    } else {
      data_valid = 1;
      clock.Step();
      data_valid = 0;
    }

    const int channel_gap = model.Next() & 3u;
    for (int cycle = 0; cycle < channel_gap; ++cycle) {
      clock.Step();
    }

    if (address_first) {
      data_valid = 1;
      clock.Step();
      data_valid = 0;
    } else {
      addr_valid = 1;
      clock.Step();
      addr_valid = 0;
    }

    Check(requests == transaction + 1,
          "random write request was not captured exactly once");
    Check(captured_addr == addr && captured_data == data[0] &&
              captured_id == id,
          "random write request fields changed");

    const int completion_delay = model.Next() & 7u;
    for (int cycle = 0; cycle < completion_delay; ++cycle) {
      clock.Step();
      Check(requests == transaction + 1,
            "random write request replayed before response");
    }

    AxiWResp response = {};
    response.write_resp_bits_id = id;
    driver.QueueResponse(response);

    int timeout = 256;
    while (driver.HasDeferredRequest() && --timeout > 0) {
      clock.Step();
    }
    Check(timeout > 0, "random write response timed out");
    Check(resp_valid == 0, "random write response was not retired");
  }

  Check(model.handshakes == 64, "random write handshake count changed");
}

}  // namespace

int
main()
{
  TestDeferredRead();
  TestIndependentWriteChannels();
  TestDmaRequestValidation();
  TestWriteBurstCollection();
  TestReadBurstResponseRetirement();
  TestErrorResponseRetirement();
  TestSingleOutstandingBackpressure();
  TestRandomizedReadTiming();
  TestRandomizedWriteTiming();
  std::puts("PASS: gem5 Coral AXI master adapter");
  return 0;
}
