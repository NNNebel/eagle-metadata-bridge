# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-04-26

### 🇬🇧 English
#### 🎉 New Features
- **Per-field output control via `config.json`**: Each tag and annotation field (checkpoint, LoRA, positive, negative, seed, steps, CFG, sampler, scheduler) can now be individually enabled or disabled in `config.json`. Changes take effect on the next execution — no ComfyUI restart required.
- **Eagle Metadata Bridge (Test) node**: A companion node with 18 individual ON/OFF toggles for every tag and annotation field. Ignores `config.json`, making it easy to preview how different settings affect the output before committing them to the file.

#### 🐛 Bug Fixes
- Fixed `scheduler` tag not being generated even when enabled in settings.
- Fixed annotation output being incorrect when all fields for a sampler step were disabled.
- Fixed edge cases where zero values for generation parameters (seed, steps, CFG) were not included in tags and annotations.

### 🇯🇵 日本語
#### 🎉 新機能
- **`config.json` によるフィールド別出力制御**: タグ・アノテーションの各フィールド（checkpoint・LoRA・positive・negative・seed・steps・CFG・sampler・scheduler）を `config.json` で個別に ON/OFF できるようになった。変更は次回実行時に反映され、ComfyUI の再起動は不要。
- **Eagle Metadata Bridge (Test) ノード**: タグ・アノテーションの全フィールドを個別にトグルできるノードを追加。`config.json` を無視するため、設定をファイルに書き込む前に出力を手軽に確認できる。

#### 🐛 バグ修正
- `scheduler` タグが設定で有効にしていても生成されていなかった不具合を修正。
- サンプラーステップのすべてのフィールドが無効の場合にアノテーション出力が不正になっていた不具合を修正。
- seed・steps・CFG が 0 のときにタグ・アノテーションに含まれないことがあった不具合を修正。

---

## [1.0.0] - 2026-04-24

### 🇬🇧 English
#### 🎉 New Features
- **Auto-tagging**: Automatically generates Eagle tags from the ComfyUI graph — checkpoint name, LoRA names, prompt tokens, seed, steps, CFG, and sampler.
- **Auto-annotation**: Writes a structured generation info block to the Eagle annotation field, including all sampler steps for multi-sampler workflows.
- **Eagle folder assignment** (`eagle_folder`): Assign images to a specific Eagle folder by name.
- **PNG and WebP support**: Save as PNG (with compression level) or WebP (with quality). Irrelevant settings are hidden per format.
- **ComfyUI preview**: Optionally show the saved image as a preview in the ComfyUI queue.
- **JPEG support**: Added JPEG as a save format option.
- **Local folder save** (`local_save_path`): Save images to any local folder path alongside Eagle. Supports `%date:yyyy-MM-dd%` / `%NodeTitle.param%` placeholders.
- **Eagle folder auto-creation** (`eagle_folder_path`): Specify a nested Eagle folder path using date and node-parameter placeholders. Folders that don't exist are created automatically.
- **Configurable Eagle port**: Set a custom Eagle API port in `config.json` (default: 41595).
- **Visible errors**: Eagle connection failures now surface as errors in the ComfyUI queue instead of silently passing.

#### ✨ Improvements
- Format-dependent widgets (`compress_level` / `quality`) are now hidden when not applicable to the selected format.
- `filename_prefix` supports placeholders (e.g. `%date:hhmmss%`).

### 🇯🇵 日本語
#### 🎉 新機能
- **自動タグ付け**: ComfyUI のグラフを解析し、チェックポイント名・LoRA 名・プロンプトトークン・seed・steps・CFG・サンプラーを Eagle タグとして自動生成。
- **自動アノテーション**: 生成情報（チェックポイント・LoRA・サンプラーごとのパラメータ）を Eagle のアノテーション欄に構造化して書き込む。マルチサンプラーワークフローにも対応。
- **Eagle フォルダ割り当て** (`eagle_folder`): フォルダ名を指定して Eagle 内の特定フォルダに画像を送れる。
- **PNG・WebP 対応**: PNG（圧縮レベル指定）または WebP（品質指定）で保存。フォーマットに応じて不要な設定は非表示。
- **ComfyUI プレビュー**: 保存した画像を ComfyUI のキューにプレビュー表示するか選択できる。
- **JPEG 対応**: 保存フォーマットに JPEG を追加。
- **ローカル保存** (`local_save_path`): Eagle への送信と同時に任意のローカルフォルダにも保存できる。`%date:yyyy-MM-dd%` / `%NodeTitle.param%` のプレースホルダーに対応。
- **Eagle フォルダ自動生成** (`eagle_folder_path`): 日付やノードパラメータのプレースホルダーを含むパスで Eagle フォルダを指定。存在しないフォルダは自動作成。
- **Eagle ポート番号の設定**: `config.json` でポート番号を変更可能（デフォルト: 41595）。
- **エラーの可視化**: Eagle への接続失敗が ComfyUI のキュー上にエラーとして表示されるようになった（以前は無音で成功扱い）。

#### ✨ 改善
- フォーマット選択に応じて不要なウィジェット（`compress_level` / `quality`）を非表示にするようになった。
- `filename_prefix` でプレースホルダー（例: `%date:hhmmss%`）が使えるようになった。
