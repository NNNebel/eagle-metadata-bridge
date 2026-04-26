"""
Annotation text generation from extracted metadata.
Corresponds to AnnotationBuilder.js on the JS side.
"""
import os


def _basename_no_ext(path):
    """Extract filename without extension (no path prefix). Same helper as tag_generator."""
    return os.path.splitext(os.path.basename(path))[0]


def _step_label(step):
    node_type = step.get("node_type", "Sampler")
    node_id   = step.get("node_id", "")
    label = f"{node_type} (ID: {node_id})" if node_id else node_type
    if step.get("is_base"):
        return f"[Base Sampler - {label}]"
    return f"[Step {step['step_index']} - {label}]"


_DEFAULT_SETTINGS = {
    "checkpoint": True,
    "lora": True,
    "positive": True,
    "negative": True,
    "seed": True,
    "steps": True,
    "cfg": True,
    "sampler": True,
    "scheduler": True,
}


def _setting(settings, key):
    if settings is None:
        return _DEFAULT_SETTINGS.get(key, True)
    return settings.get(key, _DEFAULT_SETTINGS.get(key, True))


def generate_annotation(meta, settings=None):
    """
    Generate annotation matching comfyui-auto-tagger.
    settings: dict of boolean flags controlling which fields appear.
              None means all enabled (default behaviour).
    """
    lines = ["[Generation Info]"]

    global_ckpt = None
    if _setting(settings, "checkpoint") and meta.get("checkpoint"):
        global_ckpt = _basename_no_ext(meta["checkpoint"])
        lines.append(f"Checkpoint: {global_ckpt}")

    # Always output LoRA line when loras key exists (even empty list)
    if _setting(settings, "lora") and meta.get("loras") is not None:
        lora_names = [os.path.splitext(l)[0] for l in meta["loras"]]
        lines.append("LoRA: " + ", ".join(lora_names))

    steps = meta.get("generation_steps") or []
    single_step = len(steps) == 1

    if steps:
        # Build all step blocks first; emit separator only when something will appear
        # (mirrors JS AnnotationBuilder._buildStepContent / stepBlocks logic)
        step_blocks = []
        for step in steps:
            step_lines = []

            # Show checkpoint in step when: only one step, or step ckpt differs from global
            if _setting(settings, "checkpoint"):
                step_ckpt = _basename_no_ext(step["checkpoint"]) if step.get("checkpoint") else None
                if step_ckpt and (single_step or step_ckpt != global_ckpt):
                    step_lines.append(f"Checkpoint: {step_ckpt}")

            if _setting(settings, "seed") and step.get("seed") is not None:
                step_lines.append(f"Seed: {step['seed']}")
            params = []
            if _setting(settings, "steps") and step.get("steps") is not None:
                params.append(f"Steps: {step['steps']}")
            if _setting(settings, "cfg") and step.get("cfg") is not None:
                params.append(f"CFG: {float(step['cfg']):.1f}")
            if _setting(settings, "sampler") and step.get("sampler"):
                params.append(f"Sampler: {step['sampler']}")
            if _setting(settings, "scheduler") and step.get("scheduler"):
                params.append(f"Scheduler: {step['scheduler']}")
            if params:
                step_lines.append(" | ".join(params))
            if _setting(settings, "positive") and step.get("positive"):
                step_lines.append(f"Positive: {step['positive']}")
            if _setting(settings, "negative") and step.get("negative"):
                step_lines.append(f"Negative: {step['negative']}")

            if step_lines:
                step_blocks.append([_step_label(step)] + step_lines)

        if step_blocks:
            if len(lines) > 1:  # header has content beyond [Generation Info]
                lines.append("")
            for i, block in enumerate(step_blocks):
                lines.extend(block)
                if i < len(step_blocks) - 1:
                    lines.append("")
    else:
        # Fallback: no generation_steps — build content first, emit only if non-empty
        fallback_lines = []
        if _setting(settings, "seed") and meta.get("seed") is not None:
            fallback_lines.append(f"Seed: {meta['seed']}")
        params = []
        if _setting(settings, "steps") and meta.get("steps") is not None:
            params.append(f"Steps: {meta['steps']}")
        if _setting(settings, "cfg") and meta.get("cfg") is not None:
            params.append(f"CFG: {float(meta['cfg']):.1f}")
        if _setting(settings, "sampler") and meta.get("sampler"):
            params.append(f"Sampler: {meta['sampler']}")
        if _setting(settings, "scheduler") and meta.get("scheduler"):
            params.append(f"Scheduler: {meta['scheduler']}")
        if params:
            fallback_lines.append(" | ".join(params))
        if _setting(settings, "positive") and meta.get("positive"):
            fallback_lines.extend(["", "[Positive Prompt]", meta["positive"]])
        if _setting(settings, "negative") and meta.get("negative"):
            fallback_lines.extend(["", "[Negative Prompt]", meta["negative"]])
        if fallback_lines:
            lines.append("")
            lines.extend(fallback_lines)

    return "\n".join(lines)
