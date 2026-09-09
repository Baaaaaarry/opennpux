#!/usr/bin/env python3
"""Print the machine-readable OpenNPUX backend capability contract."""

import json

from opennpux_backend import capability_manifest


def main() -> None:
    print(json.dumps(capability_manifest(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
