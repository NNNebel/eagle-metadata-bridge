import os
import json
import requests


def load_eagle_api_base():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    try:
        with open(config_path, encoding='utf-8') as f:
            port = json.load(f).get('eagle_port', 41595)
            return f"http://localhost:{port}"
    except FileNotFoundError:
        return "http://localhost:41595"
    except Exception as e:
        print(f"[EagleMetadataBridge] Failed to read config.json: {e}, using default port 41595")
        return "http://localhost:41595"


EAGLE_API_BASE = load_eagle_api_base()


def fetch_eagle_folders(api_base=None):
    """Fetch Eagle folder tree. Returns list or None on error."""
    base = api_base or EAGLE_API_BASE
    try:
        resp = requests.get(f"{base}/api/folder/list", timeout=5)
        if not resp.ok:
            print(f"[EagleMetadataBridge] Failed to fetch folder list: {resp.status_code}")
            return None
        return resp.json().get("data", [])
    except Exception as e:
        print(f"[EagleMetadataBridge] Could not fetch folder list: {e}")
        return None


def resolve_folder_id(folder_name, api_base=None):
    """Resolve a single folder name to its Eagle folder ID. Returns None if not found."""
    if not folder_name:
        return None
    folders = fetch_eagle_folders(api_base)
    if folders is None:
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


def ensure_eagle_folder_path(path_str, api_base=None):
    """
    Walk or create a nested Eagle folder path like "2D/2026-04-23/checkpoint".
    Each segment is created under the previous one if it doesn't exist.
    Returns the deepest folder's ID, or None on failure.
    """
    base = api_base or EAGLE_API_BASE
    segments = [s for s in path_str.replace('\\', '/').split('/') if s]
    if not segments:
        return None

    folders = fetch_eagle_folders(api_base)
    if folders is None:
        return None

    def find_in(nodes, name):
        for f in nodes:
            if f.get("name") == name:
                return f
        return None

    parent_id = None
    current_level = folders

    for segment in segments:
        node = find_in(current_level, segment)
        if node:
            parent_id = node["id"]
            current_level = node.get("children") or []
        else:
            body = {"folderName": segment}
            if parent_id:
                body["parent"] = parent_id
            try:
                resp = requests.post(f"{base}/api/folder/create", json=body, timeout=5)
                if not resp.ok:
                    print(f"[EagleMetadataBridge] Failed to create folder {segment!r}: {resp.status_code} {resp.text}")
                    return None
                created = resp.json().get("data", {})
                parent_id = created.get("id")
                if not parent_id:
                    print(f"[EagleMetadataBridge] No ID returned for created folder {segment!r}")
                    return None
                current_level = []
                print(f"[EagleMetadataBridge] Created Eagle folder: {segment!r} (id={parent_id})")
            except Exception as e:
                print(f"[EagleMetadataBridge] Error creating Eagle folder {segment!r}: {e}")
                return None

    return parent_id
