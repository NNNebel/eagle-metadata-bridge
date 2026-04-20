import os
import json
import numpy as np
import requests
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths


EAGLE_API_BASE = "http://localhost:41595"


def execute(images, filename_prefix, eagle_folder_id="",
            tags="", prompt=None, extra_pnginfo=None, unique_id=None):

    results = []

    for batch_idx, image_tensor in enumerate(images):
        # --- Step 1: PNG 保存 ---
        img_np = np.clip(255.0 * image_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # メタデータ構築: prompt に _eagle_final_node マーカーを注入
        pnginfo = PngInfo()
        if prompt is not None:
            pnginfo.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                pnginfo.add_text(key, json.dumps(value))
        if unique_id is not None:
            bridge_meta = {"version": 1, "final_node_id": str(unique_id)}
            pnginfo.add_text("eagle_bridge", json.dumps(bridge_meta))

        # ファイルパス決定
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, img.width, img.height
        )
        file = f"{filename}_{counter + batch_idx:05d}_.png"
        abs_path = os.path.abspath(os.path.join(full_output_folder, file))
        img.save(abs_path, pnginfo=pnginfo, compress_level=4)
        print(f"[EagleMetadataBridge] Saved: {abs_path}")

        # --- Step 2: タグ組み立て（固定値のみ / PoC） ---
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

        # --- Step 3: Eagle API 送信 ---
        payload = {
            "path": abs_path,
            "name": os.path.basename(abs_path),
            "tags": tag_list,
        }
        if eagle_folder_id:
            payload["folderId"] = eagle_folder_id

        try:
            resp = requests.post(
                f"{EAGLE_API_BASE}/api/item/addFromPath",
                json=payload,
                timeout=10,
            )
            if resp.ok:
                print(f"[EagleMetadataBridge] Sent to Eagle: {resp.json()}")
            else:
                print(f"[EagleMetadataBridge] Eagle API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"[EagleMetadataBridge] Eagle not running (port {EAGLE_API_BASE} unreachable). Image saved locally.")
        except Exception as e:
            print(f"[EagleMetadataBridge] Eagle API failed: {e}")

        results.append(abs_path)

    return {"ui": {"images": results}}
