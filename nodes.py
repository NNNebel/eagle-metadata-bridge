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
                "eagle_folder_id": ("STRING", {"default": ""}),
                "tags": ("STRING", {"default": "", "multiline": False,
                                    "tooltip": "Comma-separated fixed tags"}),
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
