"""
Tag generation from extracted metadata.
Corresponds to TagGenerator.js on the JS side.
"""
import os
import re


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
    text = text.replace("\n", ",")
    for raw in text.split(","):
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

    if meta.get("checkpoint"):
        tags.append(_basename_no_ext(meta["checkpoint"]))

    for lora in meta.get("loras") or []:
        tags.append(_basename_no_ext(lora))

    for token in _tokenize_prompt(meta.get("positive")):
        tags.append(token)

    for token in _tokenize_prompt(meta.get("negative")):
        tags.append(f"neg:{token}")

    if meta.get("seed") is not None:
        tags.append(f"seed:{meta['seed']}")
    if meta.get("steps") is not None:
        tags.append(f"steps:{meta['steps']}")
    if meta.get("cfg") is not None:
        tags.append(f"cfg:{float(meta['cfg']):.2f}")
    if meta.get("sampler"):
        tags.append(f"sampler:{str(meta['sampler']).lower()}")

    return tags
