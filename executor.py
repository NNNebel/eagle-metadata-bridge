import os
import json
import re
import numpy as np
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

from eagle.client import (
    EAGLE_API_BASE,
    fetch_eagle_folders as _fetch_eagle_folders,
    resolve_folder_id as _resolve_folder_id,
    ensure_eagle_folder_path as _ensure_eagle_folder_path,
)
from metadata_parser.graph import resolve_link as _resolve_link
from metadata_parser.comfyui_parser import extract_metadata
from metadata_parser.tag_generator import generate_tags
from metadata_parser.annotation import generate_annotation


# ---------------------------------------------------------------------------
# EXIF builder for WebP (matches ComfyUI default Save Image format)
# ---------------------------------------------------------------------------

def _build_webp_exif(entries):
    """
    Build a minimal big-endian TIFF EXIF block for WebP.
    entries: list of (tag: int, value: str) sorted by tag ascending.
    Matches ComfyUI's default WebP metadata format so existing readers work.
    Tag 0x010F (Make)  → workflow
    Tag 0x0110 (Model) → prompt
    Tag 0x013B (Artist)→ eagle_bridge
    """
    import struct
    entries = sorted(entries, key=lambda x: x[0])
    n = len(entries)
    # Layout: header(8) + count(2) + entries(12*n) + next_ifd(4) + string data
    data_offset = 8 + 2 + 12 * n + 4
    encoded = [v.encode('utf-8') + b'\x00' for _, v in entries]
    offsets = []
    pos = data_offset
    for s in encoded:
        offsets.append(pos)
        pos += len(s)

    buf = b'MM\x00\x2A\x00\x00\x00\x08'  # big-endian TIFF, IFD at offset 8
    buf += struct.pack('>H', n)
    for i, (tag, _) in enumerate(entries):
        buf += struct.pack('>HHII', tag, 2, len(encoded[i]), offsets[i])
    buf += b'\x00\x00\x00\x00'  # no next IFD
    for s in encoded:
        buf += s
    return buf


def _build_jpeg_exif(entries):
    """
    Build JPEG APP1 EXIF bytes (Exif\\0\\0 + TIFF block).
    Uses same TIFF structure as WebP; Pillow wraps it in the APP1 segment.
    Tag mapping (matches ComfyUI JPEG format):
      0x010E (ImageDescription) → Workflow
      0x010F (Make)             → Prompt
      0x013B (Artist)           → eagle_bridge
    """
    tiff = _build_webp_exif(entries)
    return b'Exif\x00\x00' + tiff




# ---------------------------------------------------------------------------
# Path expression expansion (%date:...% / %NodeTitle.param%)
# ---------------------------------------------------------------------------

def _expand_path_expr(path_str, prompt, extra_pnginfo, now=None):
    """
    Expand placeholders in a path string.
      %date:yyyy-MM-dd%  → current date (yyyy/MM/dd/HH/mm/ss supported)
      %date:hhmmss%      → current time
      %NodeTitle.param%  → value of input `param` from the node titled NodeTitle
    """
    from datetime import datetime
    if not path_str:
        return path_str
    if now is None:
        now = datetime.now()

    workflow = {}
    if extra_pnginfo and isinstance(extra_pnginfo.get('workflow'), dict):
        workflow = extra_pnginfo['workflow']

    # Build ComfyUI display-name → class_type reverse map
    # e.g. "Load Checkpoint" → "CheckpointLoaderSimple"
    display_to_class = {}
    try:
        import nodes as _comfy_nodes
        for class_type, disp in getattr(_comfy_nodes, 'NODE_DISPLAY_NAME_MAPPINGS', {}).items():
            display_to_class[disp] = class_type
    except Exception:
        pass

    # Build title → node_id map from workflow
    # Priority: explicit title > display name > class type
    title_to_id = {}
    for node in workflow.get('nodes', []):
        node_id = str(node.get('id', ''))
        if not node_id:
            continue
        class_type = node.get('type', '')
        # class type (e.g. "CheckpointLoaderSimple")
        if class_type:
            title_to_id[class_type] = node_id
        # display name (e.g. "Load Checkpoint") via NODE_DISPLAY_NAME_MAPPINGS
        try:
            import nodes as _comfy_nodes
            disp = getattr(_comfy_nodes, 'NODE_DISPLAY_NAME_MAPPINGS', {}).get(class_type, '')
            if disp:
                title_to_id[disp] = node_id
        except Exception:
            pass
        # explicit user-set title overrides all
        if node.get('title'):
            title_to_id[node['title']] = node_id

    def replace(m):
        inner = m.group(1)
        # %date:format%
        if inner.startswith('date:'):
            fmt = inner[5:]
            fmt = (fmt
                   .replace('yyyy', '%Y').replace('MM', '%m').replace('dd', '%d')
                   .replace('HH', '%H').replace('hh', '%H')
                   .replace('mm', '%M').replace('ss', '%S'))
            return now.strftime(fmt)
        # %NodeTitle.param%
        if '.' in inner:
            node_title, param_key = inner.split('.', 1)
            node_id = title_to_id.get(node_title)
            if node_id is None:
                available = list(title_to_id.keys())
                print(f"[EagleMetadataBridge] Placeholder %{inner}%: node title {node_title!r} not found. "
                      f"Available titles/types: {available}")
            elif prompt:
                val = _resolve_link(prompt, node_id, param_key)
                if val is not None:
                    # Strip directory path and extension for a clean folder name
                    return os.path.splitext(os.path.basename(str(val)))[0]
                print(f"[EagleMetadataBridge] Placeholder %{inner}%: param {param_key!r} not resolved on node {node_id}")
        return m.group(0)  # leave unexpanded if unresolvable

    expanded = re.sub(r'%([^%]+)%', replace, path_str)

    # Warn if any placeholder was not expanded
    unresolved = re.findall(r'%([^%]+)%', expanded)
    if unresolved:
        print(f"[EagleMetadataBridge] WARNING: unresolved placeholders in path: {unresolved}. "
              f"Result path: {expanded!r}")

    return expanded


# ---------------------------------------------------------------------------
# Main execute function
# ---------------------------------------------------------------------------

def execute(images, filename_prefix, eagle_folder_path="",
            tags="", format="PNG", compress_level=4, quality=85,
            preview=True, local_save_path="",
            tag_checkpoint=True, tag_lora=True, tag_positive=True, tag_negative=True,
            tag_seed=True, tag_steps=True, tag_cfg=True, tag_sampler=True, tag_scheduler=True,
            annotation_checkpoint=True, annotation_lora=True,
            annotation_positive=True, annotation_negative=True,
            annotation_seed=True, annotation_steps=True, annotation_cfg=True,
            annotation_sampler=True, annotation_scheduler=True,
            prompt=None, extra_pnginfo=None, unique_id=None):

    is_png = format == "PNG"
    is_jpeg = format == "JPEG"
    ext = ".png" if is_png else (".jpg" if is_jpeg else ".webp")
    results = []
    preview_items = []

    output_dir = folder_paths.get_output_directory()

    # Expand placeholders in filename_prefix here (prompt not yet available, use prompt arg directly)
    from datetime import datetime
    _now_pre = datetime.now()
    _expanded_prefix = _expand_path_expr(filename_prefix.strip(), prompt, extra_pnginfo, _now_pre)
    if '%' in _expanded_prefix:
        print(f"[EagleMetadataBridge] filename_prefix has unresolved placeholders: {_expanded_prefix!r}, using original.")
        _expanded_prefix = filename_prefix

    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        _expanded_prefix, output_dir, images[0].shape[1], images[0].shape[0]
    )

    # Extract metadata from graph for auto-tagging
    tag_settings = {
        "checkpoint": tag_checkpoint, "lora": tag_lora,
        "positive": tag_positive, "negative": tag_negative,
        "seed": tag_seed, "steps": tag_steps, "cfg": tag_cfg,
        "sampler": tag_sampler, "scheduler": tag_scheduler,
    }
    annotation_settings = {
        "checkpoint": annotation_checkpoint, "lora": annotation_lora,
        "positive": annotation_positive, "negative": annotation_negative,
        "seed": annotation_seed, "steps": annotation_steps, "cfg": annotation_cfg,
        "sampler": annotation_sampler, "scheduler": annotation_scheduler,
    }

    auto_meta = {}
    auto_tags = []
    auto_annotation = ""
    if prompt and unique_id:
        try:
            auto_meta = extract_metadata(prompt, unique_id)
            auto_tags = generate_tags(auto_meta, tag_settings)
            auto_annotation = generate_annotation(auto_meta, annotation_settings)
            print(f"[EagleMetadataBridge] Extracted metadata: checkpoint={auto_meta.get('checkpoint')}, "
                  f"loras={auto_meta.get('loras')}, seed={auto_meta.get('seed')}, "
                  f"steps={auto_meta.get('steps')}, cfg={auto_meta.get('cfg')}, "
                  f"sampler={auto_meta.get('sampler')}")
        except Exception as e:
            print(f"[EagleMetadataBridge] Metadata extraction failed: {e}")

    # Merge manual tags with auto-generated tags (manual tags take priority / prepended)
    manual_tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    merged_tags = manual_tag_list + [t for t in auto_tags if t not in manual_tag_list]

    # Use the same timestamp for all path expansions in this execution
    now = _now_pre
    _raw_local = local_save_path.strip()
    _expanded_local_dir = _expand_path_expr(_raw_local, prompt, extra_pnginfo, now)
    use_local_path = bool(_expanded_local_dir) and '%' not in _expanded_local_dir

    if _raw_local and not use_local_path:
        print(f"[EagleMetadataBridge] local_save_path has unresolved placeholders, skipping local save.")

    if use_local_path:
        # Relative paths are resolved from the ComfyUI OUTPUT directory
        if not os.path.isabs(_expanded_local_dir):
            _expanded_local_dir = os.path.join(output_dir, _expanded_local_dir)
        print(f"[EagleMetadataBridge] local dir: {_expanded_local_dir!r}, prefix: {_expanded_prefix!r}")

    for batch_idx, image_tensor in enumerate(images):
        img_np = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        _file_name = f"{_expanded_prefix}_{counter + batch_idx:05d}{ext}"

        # 1. Determine save location for Eagle upload
        # If use_local_path is set, save to that custom directory.
        # Otherwise, use ComfyUI's standard full_output_folder (we will delete it later).
        if use_local_path:
            try:
                os.makedirs(_expanded_local_dir, exist_ok=True)
                abs_path = os.path.abspath(os.path.join(_expanded_local_dir, _file_name))
            except Exception as e:
                print(f"[EagleMetadataBridge] Failed to create local directory: {e}")
                abs_path = os.path.abspath(os.path.join(full_output_folder, _file_name))
        else:
            abs_path = os.path.abspath(os.path.join(full_output_folder, _file_name))

        file = os.path.basename(abs_path)

        def _save_image(path):
            if is_png:
                pnginfo = PngInfo()
                if prompt is not None:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for key, value in extra_pnginfo.items():
                        pnginfo.add_text(key, json.dumps(value))
                if unique_id is not None:
                    pnginfo.add_text("eagle_bridge", json.dumps({"version": 1, "final_node_id": str(unique_id)}))
                img.save(path, pnginfo=pnginfo, compress_level=compress_level)
            elif is_jpeg:
                # JPEG: EXIF APP1 with TIFF IFD (matches ComfyUI SaveImage JPEG format)
                exif_entries = []
                workflow = (extra_pnginfo or {}).get('workflow')
                if workflow is not None:
                    exif_entries.append((0x010E, f"Workflow: {json.dumps(workflow)}"))
                if prompt is not None:
                    exif_entries.append((0x010F, f"Prompt: {json.dumps(prompt)}"))
                if unique_id is not None:
                    exif_entries.append((0x013B, f"eagle_bridge: {json.dumps({'version': 1, 'final_node_id': str(unique_id)})}"))
                exif_bytes = _build_jpeg_exif(exif_entries) if exif_entries else None
                img.save(path, format="JPEG", quality=quality, exif=exif_bytes)
            else:
                exif_entries = []
                workflow = (extra_pnginfo or {}).get('workflow')
                if workflow is not None:
                    exif_entries.append((0x010F, f"workflow: {json.dumps(workflow)}"))
                if prompt is not None:
                    exif_entries.append((0x0110, f"prompt: {json.dumps(prompt)}"))
                if unique_id is not None:
                    exif_entries.append((0x013B, f"eagle_bridge: {json.dumps({'version': 1, 'final_node_id': str(unique_id)})}"))
                exif_bytes = _build_webp_exif(exif_entries) if exif_entries else None
                img.save(path, format="WEBP", quality=quality, exif=exif_bytes)

        # Save primary image (for Eagle)
        _save_image(abs_path)
        print(f"[EagleMetadataBridge] Saved for Eagle: {abs_path}")

        # Send to Eagle
        payload = {
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "tags": merged_tags,
            "annotation": auto_annotation,
        }
        if eagle_folder_path:
            expanded_efp = _expand_path_expr(eagle_folder_path.strip(), prompt, extra_pnginfo, now)
            if '%' not in expanded_efp:
                folder_id = _ensure_eagle_folder_path(expanded_efp)
                if folder_id:
                    payload["folderId"] = folder_id

        try:
            resp = requests.post(f"{EAGLE_API_BASE}/api/item/addFromPath", json=payload, timeout=10)
            if not resp.ok:
                print(f"[EagleMetadataBridge] Eagle API error: {resp.text}")
        except Exception as e:
            print(f"[EagleMetadataBridge] Eagle connection failed: {e}")

        # 2. Handle UI Preview (Following ComfyUI standard: use temp folder)
        if preview:
            if use_local_path:
                # If we saved it permanently, we can just point to it (though UI prefers output/temp)
                # To be safe and follow convention, we use 'output' type if it's in output folder
                preview_items.append({"filename": file, "subfolder": subfolder, "type": "output"})
            else:
                # Save a copy to ComfyUI's native temp directory for UI display
                # This ensures frontend 'stat' and loading work correctly
                comfy_temp_dir = folder_paths.get_temp_directory()
                comfy_temp_path = os.path.join(comfy_temp_dir, _file_name)
                img.save(comfy_temp_path)
                preview_items.append({"filename": _file_name, "subfolder": "", "type": "temp"})

        # 3. Cleanup: If NOT saving locally, delete the temporary file in output_dir
        if not use_local_path and os.path.exists(abs_path):
            try:
                os.remove(abs_path)
                print(f"[EagleMetadataBridge] Cleaned up temporary output file: {abs_path}")
            except Exception as e:
                print(f"[EagleMetadataBridge] Failed to delete temp file {abs_path}: {e}")
    if preview:
        return {"ui": {"images": preview_items}}
    return {"ui": {"images": []}}
