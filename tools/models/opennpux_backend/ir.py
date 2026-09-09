"""Frontend-neutral OpenNPUX backend IR identities."""

GRAPH_FORMAT = "OPENNPUX_BACKEND_GRAPH_V1"
MODULE_FORMAT = "OPENNPUX_BACKEND_MODULE_V1"

# Read compatibility for artifacts emitted before the backend was separated
# from its first TVM BYOC adapter.
LEGACY_GRAPH_FORMAT = "OPENNPUX_TVM_BYOC_GRAPH_V1"
LEGACY_MODULE_FORMAT = "OPENNPUX_TVM_BYOC_MODULE_V1"

GRAPH_FORMATS = frozenset({GRAPH_FORMAT, LEGACY_GRAPH_FORMAT})
MODULE_FORMATS = frozenset({MODULE_FORMAT, LEGACY_MODULE_FORMAT})


def is_graph(value: object) -> bool:
    return isinstance(value, dict) and value.get("format") in GRAPH_FORMATS


def is_module(value: object) -> bool:
    return isinstance(value, dict) and value.get("format") in MODULE_FORMATS
