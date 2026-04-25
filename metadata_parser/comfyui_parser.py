"""
ComfyUI prompt graph parser — extracts generation metadata.
Corresponds to ComfyUIParser.js on the JS side.
"""
import os
import re

from metadata_parser.graph import bfs_distances, resolve_link


def _extract_filename(path):
    """Extract basename from a path that may use / or \\ as separator."""
    if not path:
        return path
    # Split on both forward slash and backslash
    return re.split(r'[/\\]', str(path))[-1]
from metadata_parser.sampler_analyzer import NODE_DICT, is_sampler_node, extract_sampler_step


def extract_metadata(prompt, final_node_id):
    """
    Traverse the ComfyUI prompt graph from final_node_id and extract
    generation metadata (checkpoint, LoRAs, all sampler steps).

    Returns a dict with keys:
        checkpoint, loras, generation_steps,
        seed, steps, cfg, sampler, scheduler, positive, negative
    generation_steps: list of dicts sorted by distance (closest = base sampler)
    """
    if not prompt or not final_node_id:
        return {}

    distances = bfs_distances(prompt, str(final_node_id))
    ancestors = set(distances.keys())

    meta = {
        "checkpoint": None,
        "loras": [],
        "generation_steps": [],
        "seed": None,
        "steps": None,
        "cfg": None,
        "sampler": None,
        "scheduler": None,
        "positive": None,
        "negative": None,
    }

    sampler_nodes = []

    for nid in ancestors:
        node = prompt.get(nid)
        if not node:
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs") or {}

        node_def = NODE_DICT.get(class_type, {})

        # ── Checkpoint ──────────────────────────────────────────────────
        # Dictionary-first: use explicit input_key from checkpoint_loader entries.
        # Heuristic fallback: class name contains "CheckpointLoader".
        if not meta["checkpoint"]:
            if node_def.get("type") == "checkpoint_loader":
                ckpt = resolve_link(prompt, nid, node_def["input_key"])
                if ckpt:
                    meta["checkpoint"] = _extract_filename(ckpt)
            elif not node_def and "CheckpointLoader" in class_type:
                # Heuristic fallback for unknown checkpoint loader variants
                ckpt = resolve_link(prompt, nid, "ckpt_name")
                if ckpt:
                    meta["checkpoint"] = _extract_filename(ckpt)

        # ── LoRA ────────────────────────────────────────────────────────
        # Dictionary-first: use explicit input_key from lora_loader entries.
        # Heuristic fallback: class name contains "lora" (catches stack loaders etc.)
        if node_def.get("type") == "lora_loader":
            lora = resolve_link(prompt, nid, node_def["input_key"])
            if lora:
                name = _extract_filename(lora)
                if name not in meta["loras"]:
                    meta["loras"].append(name)
        elif not node_def and "lora" in class_type.lower():
            # Heuristic fallback for unknown LoRA loader variants
            # (e.g. rgthree "Lora Loader Stack" uses lora_01, lora_02, etc.)
            for key in inputs:
                if "lora" in key.lower():
                    lora = resolve_link(prompt, nid, key)
                    if lora and isinstance(lora, str) and lora.lower().endswith(".safetensors"):
                        name = _extract_filename(lora)
                        if name not in meta["loras"]:
                            meta["loras"].append(name)

        # ── Sampler ─────────────────────────────────────────────────────
        if is_sampler_node(node):
            step = extract_sampler_step(prompt, nid, node)
            step["distance"] = distances.get(nid, 9999)
            sampler_nodes.append(step)

    # Sort samplers by distance descending: furthest upstream = base sampler (index 0)
    sampler_nodes.sort(key=lambda s: -s["distance"])
    for i, step in enumerate(sampler_nodes):
        step["is_base"] = (i == 0)
        step["step_index"] = i + 1
        step["checkpoint"] = meta["checkpoint"]

    meta["generation_steps"] = sampler_nodes

    # Populate top-level fields from base sampler
    if sampler_nodes:
        base = sampler_nodes[0]
        meta["seed"] = base["seed"]
        meta["steps"] = base["steps"]
        meta["cfg"] = base["cfg"]
        meta["sampler"] = base["sampler"]
        meta["scheduler"] = base["scheduler"]
        # Merge prompts from all samplers (mirrors JS ComfyUIParser behaviour)
        seen_pos, seen_neg = set(), set()
        all_pos, all_neg = [], []
        for step in sampler_nodes:
            p = step.get("positive")
            if p and p not in seen_pos:
                seen_pos.add(p)
                all_pos.append(p)
            n = step.get("negative")
            if n and n not in seen_neg:
                seen_neg.add(n)
                all_neg.append(n)
        meta["positive"] = "\n".join(all_pos) if all_pos else None
        meta["negative"] = "\n".join(all_neg) if all_neg else None

    return meta
