import os
import json
import re
import numpy as np
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths


EAGLE_API_BASE = "http://localhost:41595"


def _resolve_folder_id(folder_name):
    """Resolve a folder name to its Eagle folder ID. Returns None if not found."""
    if not folder_name:
        return None
    try:
        resp = requests.get(f"{EAGLE_API_BASE}/api/folder/list", timeout=5)
        if not resp.ok:
            print(f"[EagleMetadataBridge] Failed to fetch folder list: {resp.status_code}")
            return None
        folders = resp.json().get("data", [])
    except Exception as e:
        print(f"[EagleMetadataBridge] Could not fetch folder list: {e}")
        return None

    def search(nodes):
        for f in nodes:
            if f.get("name") == folder_name:
                return f["id"]
            found = search(f.get("children") or [])
            if found:
                return found
        return None

    folder_id = search(folders)
    if folder_id is None:
        print(f"[EagleMetadataBridge] Folder not found: {folder_name!r}")
    return folder_id


# ---------------------------------------------------------------------------
# Node dictionary (mirrors comfyui-auto-tagger default-dictionary.json)
# ---------------------------------------------------------------------------

def _load_node_dictionary():
    path = os.path.join(os.path.dirname(__file__), 'node_dictionary.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f).get('nodes', {})
    except Exception:
        return {}

_NODE_DICT = _load_node_dictionary()


# ---------------------------------------------------------------------------
# Graph traversal helpers
# ---------------------------------------------------------------------------

def _resolve_link(prompt, node_id, input_key, visited=None):
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
            return _resolve_link(prompt, src_id, input_key, visited)

        # Try common value keys (Primitive node etc.)
        for key in ("value", "int", "float", "text", "string",
                    "text_g", "text_l",
                    "seed", "steps", "cfg", "sampler_name", "scheduler"):
            if key in (src_node.get("inputs") or {}):
                return _resolve_link(prompt, src_id, key, visited)

    return None


def _get_ancestors(prompt, start_id):
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


def _is_sampler_node(node):
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


def _resolve_text_from_clip_node(prompt, node_id, visited=None):
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
    node_def = _NODE_DICT.get(class_type, {})

    # Provider node: extract text directly
    if node_def.get("type") == "provider":
        for key in ("text", "text_g", "text_l", "string"):
            val = inputs.get(key)
            if val is None:
                continue
            if isinstance(val, str):
                return val
            if isinstance(val, list) and len(val) == 2:
                return _resolve_link(prompt, str(val[0]), key, visited)

    # Router node: follow passthrough inputs defined in dictionary
    if node_def.get("type") == "router":
        passthrough = node_def.get("passthrough_rules", {}).get("output", [])
        parts = []
        for k in passthrough:
            val = inputs.get(k)
            if isinstance(val, list) and len(val) == 2:
                result = _resolve_text_from_clip_node(prompt, str(val[0]), visited)
                if result:
                    parts.append(result)
        return "\n".join(parts) if parts else None

    # Unknown node: try common text keys first
    for key in ("text", "text_g", "text_l", "string"):
        val = inputs.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return val
        if isinstance(val, list) and len(val) == 2:
            return _resolve_link(prompt, str(val[0]), key, visited)

    # Fallback: follow the first resolvable link
    for key, val in inputs.items():
        if isinstance(val, list) and len(val) == 2:
            result = _resolve_text_from_clip_node(prompt, str(val[0]), visited)
            if result is not None:
                return result

    return None


def _bfs_distances(prompt, start_id):
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


def _extract_sampler_step(prompt, nid, node):
    """Extract sampler parameters from a single sampler node. Returns dict or None."""
    inputs = node.get("inputs") or {}

    seed = _resolve_link(prompt, nid, "seed")
    steps = _resolve_link(prompt, nid, "steps")
    cfg = _resolve_link(prompt, nid, "cfg")

    sampler_name = _resolve_link(prompt, nid, "sampler_name")
    if sampler_name is None:
        sampler_link = inputs.get("sampler")
        if isinstance(sampler_link, list) and len(sampler_link) == 2:
            sampler_name = _resolve_link(prompt, str(sampler_link[0]), "sampler_name")

    scheduler = _resolve_link(prompt, nid, "scheduler")
    if scheduler is None:
        sched_link = inputs.get("sigmas")
        if isinstance(sched_link, list) and len(sched_link) == 2:
            scheduler = _resolve_link(prompt, str(sched_link[0]), "scheduler")

    positive = None
    pos_link = inputs.get("positive")
    if isinstance(pos_link, list) and len(pos_link) == 2:
        positive = _resolve_text_from_clip_node(prompt, str(pos_link[0]))

    negative = None
    neg_link = inputs.get("negative")
    if isinstance(neg_link, list) and len(neg_link) == 2:
        negative = _resolve_text_from_clip_node(prompt, str(neg_link[0]))

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

    distances = _bfs_distances(prompt, str(final_node_id))
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

        # ── Checkpoint ──────────────────────────────────────────────────
        if not meta["checkpoint"]:
            if "CheckpointLoader" in class_type:
                ckpt = _resolve_link(prompt, nid, "ckpt_name")
                if ckpt:
                    meta["checkpoint"] = os.path.basename(ckpt)
            elif class_type == "UNETLoader":
                unet = _resolve_link(prompt, nid, "unet_name")
                if unet:
                    meta["checkpoint"] = os.path.basename(unet)

        # ── LoRA ────────────────────────────────────────────────────────
        if "lora" in class_type.lower():
            if "LoraLoader" in class_type:
                lora = _resolve_link(prompt, nid, "lora_name")
                if lora:
                    name = os.path.basename(lora)
                    if name not in meta["loras"]:
                        meta["loras"].append(name)
            else:
                # Custom LoRA loader variants
                for key in inputs:
                    if "lora" in key.lower():
                        lora = _resolve_link(prompt, nid, key)
                        if lora and isinstance(lora, str) and lora.lower().endswith(".safetensors"):
                            name = os.path.basename(lora)
                            if name not in meta["loras"]:
                                meta["loras"].append(name)

        # ── Sampler ─────────────────────────────────────────────────────
        if _is_sampler_node(node):
            step = _extract_sampler_step(prompt, nid, node)
            step["distance"] = distances.get(nid, 9999)
            sampler_nodes.append(step)

    # Sort samplers by distance descending: furthest upstream = earliest in
    # the generation chain = base sampler (index 0)
    sampler_nodes.sort(key=lambda s: -s["distance"])
    for i, step in enumerate(sampler_nodes):
        step["is_base"] = (i == 0)
        step["step_index"] = i + 1

    meta["generation_steps"] = sampler_nodes

    # Populate top-level fields from base sampler (closest)
    if sampler_nodes:
        base = sampler_nodes[0]
        meta["seed"] = base["seed"]
        meta["steps"] = base["steps"]
        meta["cfg"] = base["cfg"]
        meta["sampler"] = base["sampler"]
        meta["scheduler"] = base["scheduler"]
        meta["positive"] = base["positive"]
        meta["negative"] = base["negative"]

    return meta


# ---------------------------------------------------------------------------
# Tag / annotation generation  (mirrors JS TagGenerator rules)
# ---------------------------------------------------------------------------

def _basename_no_ext(path):
    """Extract filename without extension, lowercase."""
    return os.path.splitext(os.path.basename(path))[0].lower()


def _tokenize_prompt(text):
    """
    Split a prompt string into individual tag strings.
    Handles comma-separated tokens, strips attention weights like (word:1.2).
    """
    if not text:
        return []
    tags = []
    # Collapse newlines into commas
    text = text.replace("\n", ",")
    for raw in text.split(","):
        # Strip attention weight syntax: (text:1.2) or [text]
        token = re.sub(r"[\(\[\{]|[\)\]\}]|:[0-9.]+", "", raw).strip()
        if token:
            tags.append(token)
    return tags


def generate_tags(meta):
    """
    Generate Eagle tags from extracted metadata.
    Mirrors the JS TagGenerator.generate() logic with all options enabled.
    Returns a list of tag strings.
    """
    tags = []

    # Checkpoint
    if meta.get("checkpoint"):
        tags.append(_basename_no_ext(meta["checkpoint"]))

    # LoRAs
    for lora in meta.get("loras") or []:
        tags.append(_basename_no_ext(lora))

    # Positive prompt tokens
    for token in _tokenize_prompt(meta.get("positive")):
        tags.append(token)

    # Negative prompt tokens  (neg: prefix)
    for token in _tokenize_prompt(meta.get("negative")):
        tags.append(f"neg:{token}")

    # Parameters
    if meta.get("seed") is not None:
        tags.append(f"seed:{meta['seed']}")
    if meta.get("steps") is not None:
        tags.append(f"steps:{meta['steps']}")
    if meta.get("cfg") is not None:
        tags.append(f"cfg:{float(meta['cfg']):.2f}")
    if meta.get("sampler"):
        tags.append(f"sampler:{str(meta['sampler']).lower()}")

    return tags


def _step_label(step):
    node_type = step.get("node_type", "Sampler")
    if step.get("is_base"):
        return f"[Base Sampler - {node_type}]"
    return f"[Step {step['step_index']} - {node_type}]"


def generate_annotation(meta):
    """
    Generate annotation matching comfyui-auto-tagger with all output settings enabled.
    Format:
      [Generation Info]
      Checkpoint: <name without ext>
      LoRA: <name without ext>, ...

      [Base Sampler - KSampler]
      Seed: <value>
      Steps: <N> | CFG: <N> | Sampler: <name> | Scheduler: <name>
      Positive: <text>
      Negative: <text>

      [Step 2 - KSampler]
      ...
    """
    lines = ["[Generation Info]"]

    if meta.get("checkpoint"):
        lines.append(f"Checkpoint: {os.path.splitext(meta['checkpoint'])[0]}")

    if meta.get("loras"):
        lora_names = [os.path.splitext(l)[0] for l in meta["loras"]]
        lines.append("LoRA: " + ", ".join(lora_names))

    steps = meta.get("generation_steps") or []

    if steps:
        for step in steps:
            lines.append("")
            lines.append(_step_label(step))
            if step.get("seed") is not None:
                lines.append(f"Seed: {step['seed']}")
            params = []
            if step.get("steps") is not None:
                params.append(f"Steps: {step['steps']}")
            if step.get("cfg") is not None:
                params.append(f"CFG: {float(step['cfg']):.2f}")
            if step.get("sampler"):
                params.append(f"Sampler: {step['sampler']}")
            if step.get("scheduler"):
                params.append(f"Scheduler: {step['scheduler']}")
            if params:
                lines.append(" | ".join(params))
            if step.get("positive"):
                lines.append(f"Positive: {step['positive']}")
            if step.get("negative"):
                lines.append(f"Negative: {step['negative']}")
    else:
        # Fallback: no generation_steps (single sampler legacy path)
        lines.append("")
        if meta.get("seed") is not None:
            lines.append(f"Seed: {meta['seed']}")
        params = []
        if meta.get("steps") is not None:
            params.append(f"Steps: {meta['steps']}")
        if meta.get("cfg") is not None:
            params.append(f"CFG: {float(meta['cfg']):.2f}")
        if meta.get("sampler"):
            params.append(f"Sampler: {meta['sampler']}")
        if meta.get("scheduler"):
            params.append(f"Scheduler: {meta['scheduler']}")
        if params:
            lines.append(" | ".join(params))
        if meta.get("positive"):
            lines.append(f"Positive: {meta['positive']}")
        if meta.get("negative"):
            lines.append(f"Negative: {meta['negative']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main execute function
# ---------------------------------------------------------------------------

def execute(images, filename_prefix, eagle_folder="",
            tags="", format="PNG", compress_level=4, quality=85,
            preview=True, prompt=None, extra_pnginfo=None, unique_id=None):

    is_png = format == "PNG"
    ext = ".png" if is_png else ".webp"
    results = []
    preview_items = []

    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )

    # Extract metadata from graph for auto-tagging
    auto_meta = {}
    auto_tags = []
    auto_annotation = ""
    if prompt and unique_id:
        try:
            auto_meta = extract_metadata(prompt, unique_id)
            auto_tags = generate_tags(auto_meta)
            auto_annotation = generate_annotation(auto_meta)
            print(f"[EagleMetadataBridge] Extracted metadata: checkpoint={auto_meta.get('checkpoint')}, "
                  f"loras={auto_meta.get('loras')}, seed={auto_meta.get('seed')}, "
                  f"steps={auto_meta.get('steps')}, cfg={auto_meta.get('cfg')}, "
                  f"sampler={auto_meta.get('sampler')}")
        except Exception as e:
            print(f"[EagleMetadataBridge] Metadata extraction failed: {e}")

    # Merge manual tags with auto-generated tags (manual tags take priority / prepended)
    manual_tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    merged_tags = manual_tag_list + [t for t in auto_tags if t not in manual_tag_list]

    for batch_idx, image_tensor in enumerate(images):
        img_np = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        file = f"{filename}_{counter + batch_idx:05d}_{ext}"
        abs_path = os.path.abspath(os.path.join(full_output_folder, file))

        if is_png:
            pnginfo = PngInfo()
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    pnginfo.add_text(key, json.dumps(value))
            if unique_id is not None:
                pnginfo.add_text("eagle_bridge", json.dumps({"version": 1, "final_node_id": str(unique_id)}))
            img.save(abs_path, pnginfo=pnginfo, compress_level=compress_level)
        else:
            xmp_parts = []
            if prompt is not None:
                xmp_parts.append(f"prompt: {json.dumps(prompt)}")
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    xmp_parts.append(f"{key}: {json.dumps(value)}")
            if unique_id is not None:
                xmp_parts.append(f"eagle_bridge: {json.dumps({'version': 1, 'final_node_id': str(unique_id)})}")
            xmp = "\n".join(xmp_parts).encode("utf-8")
            img.save(abs_path, format="WEBP", quality=quality, xmp=xmp)

        print(f"[EagleMetadataBridge] Saved: {abs_path}")

        # Send to Eagle
        payload = {
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "tags": merged_tags,
            "annotation": auto_annotation,
        }
        if eagle_folder:
            folder_id = _resolve_folder_id(eagle_folder)
            if folder_id:
                payload["folderId"] = folder_id

        try:
            resp = requests.post(f"{EAGLE_API_BASE}/api/item/addFromPath", json=payload, timeout=10)
            if resp.ok:
                print(f"[EagleMetadataBridge] Sent to Eagle: {resp.json()}")
            else:
                print(f"[EagleMetadataBridge] Eagle API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"[EagleMetadataBridge] Eagle not running. Image saved locally.")
        except Exception as e:
            print(f"[EagleMetadataBridge] Eagle API failed: {e}")

        preview_items.append({"filename": file, "subfolder": subfolder, "type": "output"})

    if preview:
        return {"ui": {"images": preview_items}}
    return {"ui": {"images": []}}
