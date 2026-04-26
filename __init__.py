import sys
import os

# Ensure eagle/ and metadata_parser/ packages are importable when loaded
# by ComfyUI's importlib-based custom node loader (which does not add the
# node directory to sys.path automatically).
_node_dir = os.path.dirname(os.path.abspath(__file__))
if _node_dir not in sys.path:
    sys.path.insert(0, _node_dir)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__version__ = "1.1.0"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
