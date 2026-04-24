#!/usr/bin/env node
/**
 * generate_bridge_expected.js
 *
 * Reads each bridge-*.json from comfyui-auto-tagger/tests/expected/,
 * runs JS TagGenerator and AnnotationBuilder with "all settings enabled",
 * and writes `tags` + `annotation` back into the JSON.
 *
 * Usage:
 *   node scripts/generate_bridge_expected.js [--cat-path PATH]
 *
 * --cat-path  Path to comfyui-auto-tagger (default: ../../comfyui-auto-tagger)
 */
'use strict';

const fs   = require('fs');
const path = require('path');

// Parse --cat-path argument
let catPath = path.join(__dirname, '..', '..', 'comfyui-auto-tagger');
for (let i = 2; i < process.argv.length; i++) {
  if (process.argv[i] === '--cat-path' && process.argv[i + 1]) {
    catPath = process.argv[++i];
  }
}

const TagGenerator    = require(path.join(catPath, 'js/metadata-processor/TagGenerator'));
const AnnotationBuilder = require(path.join(catPath, 'js/metadata-processor/AnnotationBuilder'));

const SETTINGS = {
  checkpoint: true,
  lora:       true,
  positive:   true,
  negative:   true,
  seed:       true,
  steps:      true,
  cfg:        true,
  sampler:    true,
  scheduler:  true,
  includeAllSamplers: false,
};

// Minimal translation stub — returns the key suffix as the label
function t(key) {
  const labels = {
    'ui.option.checkpoint': 'Checkpoint',
    'ui.option.lora':       'LoRA',
    'ui.option.seed':       'Seed',
    'ui.option.steps':      'Steps',
    'ui.option.sampler':    'Sampler',
    'log.caution.sampler_fallback': 'Base sampler detected heuristically',
  };
  return labels[key] || key;
}

const expectedDir = path.join(catPath, 'tests', 'expected');
const files = fs.readdirSync(expectedDir).filter(f => f.startsWith('bridge-') && f.endsWith('.json'));

for (const file of files) {
  const filePath = path.join(expectedDir, file);
  const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));

  if (!data.generationSteps) {
    console.log(`SKIP ${file} (no generationSteps)`);
    continue;
  }

  const { tags } = TagGenerator.generate(data, SETTINGS);
  const annotation = AnnotationBuilder.build(data, SETTINGS, t);

  data.tags       = [...tags].sort();
  data.annotation = annotation;

  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n', 'utf8');
  console.log(`OK   ${file}  (${tags.size} tags)`);
}
