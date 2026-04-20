import importlib
from . import executor


class EagleMetadataBridge:
    RETURN_TYPES = ()
    FUNCTION = "send_to_eagle"
    OUTPUT_NODE = True
    CATEGORY = "Eagle"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {
                "eagle_folder": ("STRING", {"default": "", "tooltip": "Eagle folder name (e.g. 'ComfyUI outputs')"}),
                "tags": ("STRING", {"default": "", "multiline": False,
                                    "tooltip": "Comma-separated fixed tags"}),
                "format": (["PNG", "WebP"],),
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1,
                                           "tooltip": "PNG compression level (0=none, 9=max)"}),
                "quality": ("INT", {"default": 85, "min": 1, "max": 100, "step": 1,
                                    "tooltip": "WebP quality (1-100)"}),
                "preview": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    def send_to_eagle(self, **kwargs):
        importlib.reload(executor)
        return executor.execute(**kwargs)


NODE_CLASS_MAPPINGS = {
    "EagleMetadataBridge": EagleMetadataBridge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleMetadataBridge": "Eagle Metadata Bridge",
}
