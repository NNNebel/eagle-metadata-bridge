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
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "ComfyUI",
                        "tooltip": (
                            "ファイル名のプレフィックス。%date:hhmmss% などのプレースホルダー使用可。"
                            " 例: %date:hhmmss%_suffix → <時刻>_suffix_00001.png"
                        ),
                    },
                ),
            },
            "optional": {
                "eagle_folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Eagle の保存先フォルダをパスで指定。存在しないフォルダは自動作成。"
                            " プレースホルダー使用可。"
                            " 例: カテゴリ/%date:yyyy-MM-dd%/%NodeTitle.param%"
                        ),
                    },
                ),
                "tags": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": "Eagle に付与する固定タグ（カンマ区切り）。自動タグと合算される。 例: fav,WIP",
                    },
                ),
                "format": (["PNG", "WebP", "JPEG"],),
                "compress_level": (
                    "INT",
                    {
                        "default": 4, "min": 0, "max": 9, "step": 1,
                        "tooltip": "PNG 圧縮レベル（0=無圧縮/高速, 9=最大圧縮/低速）。format=PNG のときのみ有効。",
                    },
                ),
                "quality": (
                    "INT",
                    {
                        "default": 85, "min": 1, "max": 100, "step": 1,
                        "tooltip": "WebP / JPEG の品質（1〜100）。format=WebP または JPEG のときのみ有効。",
                    },
                ),
                "preview": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "True: ComfyUI キューに画像プレビューを表示する。",
                    },
                ),
                "local_save_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "ローカル保存先ディレクトリ。filename_prefix と合成してファイルパスになる。"
                            " 相対パスは ComfyUI OUTPUT フォルダ基点、絶対パスも可。プレースホルダー使用可。"
                            " 例: カテゴリ/%date:yyyy-MM-dd%/%NodeTitle.param%"
                        ),
                    },
                ),
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
