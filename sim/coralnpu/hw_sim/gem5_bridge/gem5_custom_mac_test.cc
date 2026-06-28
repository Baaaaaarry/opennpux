#include "hw_sim/gem5_bridge/gem5_custom_mac.h"

#include <cstdlib>
#include <iostream>

namespace {

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
  }
}

}  // namespace

int main() {
  VerilatedContext context;
  Gem5CustomMac mac(&context);

  Check(mac.Read32(Gem5CustomMac::kBase + 0x1c) == 0x4e5058a1,
        "custom accelerator ID mismatch");
  mac.Write32(Gem5CustomMac::kBase + 0x00, 7);
  mac.Write32(Gem5CustomMac::kBase + 0x04, 1);
  mac.Write32(Gem5CustomMac::kBase + 0x08, 35);
  mac.Write32(Gem5CustomMac::kBase + 0x0c, 1);
  for (int i = 0; i < 4 && (mac.Read32(Gem5CustomMac::kBase + 0x14) & 1) == 0;
       ++i) {
    mac.Step();
  }

  Check(mac.Read32(Gem5CustomMac::kBase + 0x10) == 42,
        "custom MAC result mismatch");
  Check((mac.Read32(Gem5CustomMac::kBase + 0x14) & 1) != 0,
        "custom MAC did not complete");
  Check(mac.Read32(Gem5CustomMac::kBase + 0x18) == 3,
        "custom MAC cycle count mismatch");
  std::cout << "PASS: custom MAC RTL" << std::endl;
  return 0;
}
