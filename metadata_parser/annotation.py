"""
Annotation text generation from extracted metadata.
Corresponds to AnnotationBuilder.js on the JS side.
"""
import os


def _step_label(step):
    node_type = step.get("node_type", "Sampler")
    if step.get("is_base"):
        return f"[Base Sampler - {node_type}]"
    return f"[Step {step['step_index']} - {node_type}]"


def generate_annotation(meta):
    """
    Generate annotation matching comfyui-auto-tagger with all output settings enabled.
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
