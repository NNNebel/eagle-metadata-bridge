"""
Usage: python check_metadata.py <path_to_png>
       python check_metadata.py  (uses latest file in C:\ComfyUI\output)
"""
import sys
import json
import os
from PIL import Image


def check(path):
    img = Image.open(path)
    info = img.info  # PNG tEXt chunks

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Chunks found: {list(info.keys())}\n")

    if "prompt" not in info:
        print("❌ 'prompt' chunk missing")
        return

    try:
        prompt = json.loads(info["prompt"])
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse prompt JSON: {e}")
        return

    print(f"Total nodes in prompt: {len(prompt)}")

    # eagle_bridge chunk (new protocol)
    eb_raw = info.get("eagle_bridge")
    if eb_raw:
        try:
            eb = json.loads(eb_raw)
            print(f"\n✅ eagle_bridge = {eb}")
        except json.JSONDecodeError:
            print(f"\n❌ eagle_bridge chunk invalid JSON: {eb_raw!r}")
    else:
        print("\n❌ eagle_bridge chunk missing")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        output_dir = r"C:\ComfyUI\output"
        pngs = sorted(
            [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")],
            key=os.path.getmtime
        )
        if not pngs:
            print("No PNG files found in output directory")
            sys.exit(1)
        target = pngs[-1]
        print(f"(Auto-selected latest: {target})")

    check(target)
