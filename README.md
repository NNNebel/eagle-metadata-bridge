# Eagle Metadata Bridge

A [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node that saves generated images directly to [Eagle](https://en.eagle.cool/) with automatically extracted metadata — checkpoint, LoRAs, prompt, seed, sampler settings — attached as tags and annotations.

![Eagle Metadata Bridge node](assets/node_screenshot.png)

## Features

- **Auto-tagging** — Reads the ComfyUI workflow graph and generates Eagle tags: checkpoint name, LoRA names, prompt tokens, seed, steps, CFG, sampler
- **Auto-annotation** — Writes a structured generation info block to the Eagle annotation field; supports multi-sampler workflows (e.g. hires.fix)
- **PNG / WebP / JPEG** — Choose your output format; quality/compression settings hide automatically when not applicable
- **Dynamic folder assignment** — Route images into Eagle folders using date and node-parameter placeholders (e.g. `Portraits/%date:yyyy-MM-dd%`)
- **Local save** — Optionally save a copy to a local directory at the same time
- **Visible errors** — Eagle connection failures appear as errors in the ComfyUI queue instead of silently failing

### Companion plugin

Install [comfyui-auto-tagger](https://github.com/NNNebel/comfyui-auto-tagger) in Eagle to read the embedded metadata and display full generation info inside Eagle.

## Requirements

- [Eagle](https://en.eagle.cool/) desktop app running (version 4.0+ recommended)
- ComfyUI

## Installation

### Via ComfyUI-Manager (recommended)

Search for **Eagle Metadata Bridge** in the ComfyUI-Manager custom node list and click Install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/NNNebel/eagle-metadata-bridge.git
cd eagle-metadata-bridge
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Usage

Add the **Eagle Metadata Bridge** node (category: **Eagle**) as the final output node in your workflow instead of — or alongside — Save Image.

Connect your image tensor to `images` and queue the prompt. The node will:
1. Save the image with embedded metadata
2. Send it to Eagle with auto-generated tags and annotation

### Parameters

| Parameter | Description |
|-----------|-------------|
| `images` | Image tensor from your pipeline |
| `filename_prefix` | Filename prefix. Supports placeholders: `%date:hhmmss%`, `%NodeTitle.param%` |
| `eagle_folder_path` | Nested Eagle folder path (auto-created if missing). Supports placeholders. Example: `Characters/%date:yyyy-MM-dd%` |
| `tags` | Manual tags to add (comma-separated). Combined with auto-generated tags |
| `format` | Output format: `PNG`, `WebP`, or `JPEG` |
| `compress_level` | PNG compression level 0–9 (visible only when format = PNG) |
| `quality` | WebP/JPEG quality 1–100 (visible only when format = WebP or JPEG) |
| `preview` | Show image preview in the ComfyUI queue |
| `local_save_path` | Also save to this local directory. Relative paths use the ComfyUI output folder as base. Supports placeholders |

### Filename / folder placeholders

| Placeholder | Example output |
|-------------|---------------|
| `%date:yyyy-MM-dd%` | `2026-04-24` |
| `%date:hhmmss%` | `153012` |
| `%NodeTitle.param%` | value of `param` input on a node titled `NodeTitle` |

### Eagle port

If Eagle is running on a non-default port, edit `config.json`:

```json
{
  "eagle_port": 41595
}
```

## Sample workflow

`examples/sample-workflow.webp` contains embedded workflow metadata.  
Drag and drop it onto the ComfyUI canvas to load the workflow directly.

The workflow demonstrates:
- Eagle Metadata Bridge as the output node
- ADetailer-style img2img pass (DetailerForEachDebug)
- Automatic tag and annotation generation

## License

MIT — see [LICENSE](LICENSE)
