import os
import json
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

        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        payload = {
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "tags": tag_list,
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
