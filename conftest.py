"""
conftest.py — pytest root configuration.

The root __init__.py does `from .nodes import ...` which requires a ComfyUI
runtime. Stub it out so pytest can load the package without ComfyUI installed.
"""
import sys
import types

# Stub ComfyUI-only modules before __init__.py is imported as a package init
for _mod in ('folder_paths',):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Stub nodes module so `from .nodes import ...` in __init__.py succeeds
_nodes_stub = types.ModuleType('eagle_metadata_bridge.nodes')
_nodes_stub.NODE_CLASS_MAPPINGS = {}
_nodes_stub.NODE_DISPLAY_NAME_MAPPINGS = {}
sys.modules['eagle_metadata_bridge.nodes'] = _nodes_stub
sys.modules['nodes'] = _nodes_stub
