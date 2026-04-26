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


_DEFAULT_SETTINGS = {
    "checkpoint": True,
    "lora": True,
    "positive": True,
    "negative": True,
    "seed": True,
    "steps": True,
    "cfg": True,
    "sampler": True,
    # Note: "scheduler" is intentionally absent — scheduler is not output as a tag
    # (matches JS TagGenerator behaviour). It is valid in annotation settings only.
    "include_all_samplers": False,
}


def _setting(settings, key):
    if settings is None:
        return _DEFAULT_SETTINGS.get(key, True)
    return settings.get(key, _DEFAULT_SETTINGS.get(key, True))


def generate_tags(meta, settings=None):
    """
    Generate Eagle tags from extracted metadata.
    Mirrors the JS TagGenerator.generate() logic.
    settings: dict of boolean flags (keys: checkpoint, lora, positive, negative,
              seed, steps, cfg, sampler, scheduler, include_all_samplers).
              None means all enabled (default behaviour).
    Returns a list of tag strings.
    """
    tags = []

    if _setting(settings, "checkpoint") and meta.get("checkpoint"):
        tags.append(_basename_no_ext(meta["checkpoint"]))

    if _setting(settings, "lora"):
        for lora in meta.get("loras") or []:
            tags.append(_basename_no_ext(lora))

    if _setting(settings, "positive"):
        for token in _tokenize_prompt(meta.get("positive")):
            tags.append(token)

    if _setting(settings, "negative"):
        for token in _tokenize_prompt(meta.get("negative")):
            tags.append(f"neg:{token}")

    if _setting(settings, "seed") and meta.get("seed") is not None:
        tags.append(f"seed:{meta['seed']}")
    if _setting(settings, "steps") and meta.get("steps") is not None:
        tags.append(f"steps:{meta['steps']}")
    if _setting(settings, "cfg") and meta.get("cfg") is not None:
        tags.append(f"cfg:{float(meta['cfg']):.2f}")
    if _setting(settings, "sampler") and meta.get("sampler"):
        tags.append(f"sampler:{str(meta['sampler']).lower()}")

    return tags
