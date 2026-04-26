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


class EagleMetadataBridgeTest:
    """
    テスト用ノード。config.json を無視して BOOLEAN パラメータを直接使う。
    「この設定にしたらどういうタグ・アノテーションになるか」を試すためのノード。
    設定が決まったら config.json に書き移して通常ノードに戻すことを推奨。
    """
    RETURN_TYPES = ()
    FUNCTION = "send_to_eagle"
    OUTPUT_NODE = True
    CATEGORY = "Eagle"

    @classmethod
    def INPUT_TYPES(cls):
        base = EagleMetadataBridge.INPUT_TYPES()
        optional = dict(base["optional"])
        optional.update({
            # --- Tag output settings (scheduler is annotation-only; not listed here) ---
            "tag_checkpoint": ("BOOLEAN", {"default": True, "tooltip": "タグ: チェックポイント名を含める。"}),
            "tag_lora": ("BOOLEAN", {"default": True, "tooltip": "タグ: LoRA 名を含める。"}),
            "tag_positive": ("BOOLEAN", {"default": True, "tooltip": "タグ: ポジティブプロンプトのトークンを含める。"}),
            "tag_negative": ("BOOLEAN", {"default": True, "tooltip": "タグ: ネガティブプロンプトのトークンを含める（neg: プレフィックス付き）。"}),
            "tag_seed": ("BOOLEAN", {"default": True, "tooltip": "タグ: seed 値を含める（seed:NNNN）。"}),
            "tag_steps": ("BOOLEAN", {"default": True, "tooltip": "タグ: ステップ数を含める（steps:NN）。"}),
            "tag_cfg": ("BOOLEAN", {"default": True, "tooltip": "タグ: CFG スケールを含める（cfg:N.NN）。"}),
            "tag_sampler": ("BOOLEAN", {"default": True, "tooltip": "タグ: サンプラー名を含める（sampler:name）。"}),
            # --- Annotation output settings ---
            "annotation_checkpoint": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: チェックポイント行を出力する。"}),
            "annotation_lora": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: LoRA 行を出力する。"}),
            "annotation_positive": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Positive プロンプト行を出力する。"}),
            "annotation_negative": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Negative プロンプト行を出力する。"}),
            "annotation_seed": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Seed 行を出力する。"}),
            "annotation_steps": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Steps を出力する。"}),
            "annotation_cfg": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: CFG を出力する。"}),
            "annotation_sampler": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Sampler を出力する。"}),
            "annotation_scheduler": ("BOOLEAN", {"default": True, "tooltip": "アノテーション: Scheduler を出力する。"}),
        })
        return {**base, "optional": optional}

    def send_to_eagle(self, **kwargs):
        importlib.reload(executor)
        # BOOLEAN パラメータから settings dict を組み立て、config.json を上書き
        # scheduler はアノテーション専用（タグとしては出力されないため tag_keys には含めない）
        _tag_keys = ["checkpoint", "lora", "positive", "negative",
                     "seed", "steps", "cfg", "sampler"]
        _ann_keys = ["checkpoint", "lora", "positive", "negative",
                     "seed", "steps", "cfg", "sampler", "scheduler"]
        tag_settings = {k: kwargs.pop(f"tag_{k}", True) for k in _tag_keys}
        ann_settings = {k: kwargs.pop(f"annotation_{k}", True) for k in _ann_keys}
        return executor.execute(
            _settings_override_tag=tag_settings,
            _settings_override_annotation=ann_settings,
            **kwargs,
        )


NODE_CLASS_MAPPINGS = {
    "EagleMetadataBridge": EagleMetadataBridge,
    "EagleMetadataBridgeTest": EagleMetadataBridgeTest,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EagleMetadataBridge": "Eagle Metadata Bridge",
    "EagleMetadataBridgeTest": "Eagle Metadata Bridge (Test)",
}
