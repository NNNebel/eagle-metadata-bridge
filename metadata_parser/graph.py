"""
Graph traversal utilities for ComfyUI prompt graphs.
Corresponds to ComfyUIGraph.js on the JS side.
"""


def resolve_link(prompt, node_id, input_key, visited=None):
    """
    Resolve an input value, following node-to-node links recursively.
    Returns the scalar value or None if unresolvable.
    Link format in ComfyUI prompt: [source_node_id, output_slot]
    """
    if visited is None:
        visited = set()
    if node_id in visited:
        return None
    visited = visited | {node_id}

    node = prompt.get(str(node_id))
    if not node or not node.get("inputs"):
        return None

    value = node["inputs"].get(input_key)
    if value is None:
        return None

    # Direct scalar value
    if not isinstance(value, list):
        return value

    # Link: [source_node_id, output_slot]
    if len(value) == 2:
        src_id = str(value[0])
        src_node = prompt.get(src_id)
        if not src_node:
            return None

        # Try same key (passthrough pattern)
        if input_key in (src_node.get("inputs") or {}):
            return resolve_link(prompt, src_id, input_key, visited)

        # Try common value keys (Primitive node etc.)
        for key in ("value", "int", "float", "text", "string",
                    "text_g", "text_l",
                    "seed", "steps", "cfg", "sampler_name", "scheduler"):
            if key in (src_node.get("inputs") or {}):
                return resolve_link(prompt, src_id, key, visited)

    return None


def get_ancestors(prompt, start_id):
    """BFS: return set of all ancestor node IDs reachable from start_id."""
    visited = set()
    queue = [str(start_id)]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        node = prompt.get(nid)
        if not node:
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2:
                parent_id = str(val[0])
                if parent_id not in visited:
                    queue.append(parent_id)
    return visited


def bfs_distances(prompt, start_id):
    """BFS from start_id; return {node_id: distance} for all reachable ancestors."""
    distances = {}
    queue = [(str(start_id), 0)]
    while queue:
        nid, dist = queue.pop(0)
        if nid in distances:
            continue
        distances[nid] = dist
        node = prompt.get(nid)
        if not node:
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, list) and len(val) == 2:
                parent_id = str(val[0])
                if parent_id not in distances:
                    queue.append((parent_id, dist + 1))
    return distances
