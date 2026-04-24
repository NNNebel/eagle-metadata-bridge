"""
conftest.py — pytest configuration for the tests/ suite.

Stubs out ComfyUI-only runtime modules so executor.py can be imported
without a live ComfyUI installation.
"""
import sys
import os
import types

# Stub ComfyUI-only modules before executor.py is imported
for _mod in ('folder_paths',):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

# Stub nodes module so any `from .nodes import ...` in __init__.py succeeds
_nodes_stub = types.ModuleType('eagle_metadata_bridge.nodes')
_nodes_stub.NODE_CLASS_MAPPINGS = {}
_nodes_stub.NODE_DISPLAY_NAME_MAPPINGS = {}
sys.modules['eagle_metadata_bridge.nodes'] = _nodes_stub
sys.modules['nodes'] = _nodes_stub
