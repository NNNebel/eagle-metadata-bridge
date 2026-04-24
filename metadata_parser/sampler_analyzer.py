"""
Sampler node detection and parameter extraction.
Corresponds to ComfyUISamplerAnalyzer.js on the JS side.
"""
import os
import json

from metadata_parser.graph import resolve_link


def _load_node_dictionary():
    path = os.path.join(os.path.dirname(__file__), '..', 'node_dictionary.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('nodes', {})
    except Exception:
        return {}


NODE_DICT = _load_node_dictionary()


def is_sampler_node(node):
    """Return True if the node looks like a sampler."""
    inputs = node.get("inputs") or {}
    keys = set(inputs.keys())
    # Traditional: seed + steps + cfg + (positive or negative)
    trad = {"seed", "steps", "cfg"}
    if trad <= keys and ("positive" in keys or "negative" in keys):
        return True
    # Advanced: sampler + sigmas + latent_image
    if {"sampler", "sigmas", "latent_image"} <= keys:
        return True
    return False


def resolve_text_from_clip_node(prompt, node_id, visited=None):
    """
    Follow a conditioning link to a CLIPTextEncode-style node and
    return its text value.
    """
    if visited is None:
        visited = set()
    if node_id in visited:
        return None
    visited = visited | {node_id}

    node = prompt.get(str(node_id))
    if not node:
        return None

    inputs = node.get("inputs") or {}
    class_type = node.get("class_type", "")
    node_def = NODE_DICT.get(class_type, {})

    # Provider node: extract text directly
    if node_def.get("type") == "provider":
        for key in ("text", "text_g", "text_l", "string"):
            val = inputs.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                return val
            if isinstance(val, list) and len(val) == 2:
                return resolve_link(prompt, str(val[0]), key, visited)

    # Router node: follow passthrough inputs defined in dictionary
    if node_def.get("type") == "router":
        passthrough = node_def.get("passthrough_rules", {}).get("output", [])
        parts = []
        for k in passthrough:
            val = inputs.get(k)
            if isinstance(val, list) and len(val) == 2:
                result = resolve_text_from_clip_node(prompt, str(val[0]), visited)
                if result and result.strip():
                    parts.append(result.strip())
        return ", ".join(parts) if parts else None

    # Unknown node: try common text keys first
    for key in ("text", "text_g", "text_l", "string"):
        val = inputs.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return val
        if isinstance(val, list) and len(val) == 2:
            return resolve_link(prompt, str(val[0]), key, visited)

    # Fallback: follow the first resolvable link
    for key, val in inputs.items():
        if isinstance(val, list) and len(val) == 2:
            result = resolve_text_from_clip_node(prompt, str(val[0]), visited)
            if result is not None:
                return result

    return None


def extract_sampler_step(prompt, nid, node):
    """Extract sampler parameters from a single sampler node. Returns dict or None."""
    inputs = node.get("inputs") or {}

    seed = resolve_link(prompt, nid, "seed")
    steps = resolve_link(prompt, nid, "steps")
    cfg = resolve_link(prompt, nid, "cfg")

    sampler_name = resolve_link(prompt, nid, "sampler_name")
    if sampler_name is None:
        sampler_link = inputs.get("sampler")
        if isinstance(sampler_link, list) and len(sampler_link) == 2:
            sampler_name = resolve_link(prompt, str(sampler_link[0]), "sampler_name")

    scheduler = resolve_link(prompt, nid, "scheduler")
    if scheduler is None:
        sched_link = inputs.get("sigmas")
        if isinstance(sched_link, list) and len(sched_link) == 2:
            scheduler = resolve_link(prompt, str(sched_link[0]), "scheduler")

    positive = None
    pos_link = inputs.get("positive")
    if isinstance(pos_link, list) and len(pos_link) == 2:
        t = resolve_text_from_clip_node(prompt, str(pos_link[0]))
        positive = t.strip() if t is not None else None

    negative = None
    neg_link = inputs.get("negative")
    if isinstance(neg_link, list) and len(neg_link) == 2:
        t = resolve_text_from_clip_node(prompt, str(neg_link[0]))
        negative = t.strip() if t is not None else None

    return {
        "node_id": nid,
        "node_type": node.get("class_type", "Sampler"),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler_name,
        "scheduler": scheduler,
        "positive": positive,
        "negative": negative,
    }
