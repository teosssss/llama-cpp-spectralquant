#!/usr/bin/env bash
set -u

mkdir -p calibration_data experiments/turbo_pca_wiki

MODELS=(
  "models/SmolLM2-360M-Instruct.Q4_K_M.gguf"
  "models/qwen2-0_5b-instruct-q4_k_m.gguf"
  "models/gemma-3-1b-it-Q4_K_M.gguf"
  "models/qwen2-1_5b-instruct-q4_k_m.gguf"
  "models/Qwen3-1.7B-Q4_K_M.gguf"
)

CACHE_TYPES=(
  "turbo3_pca"
  "turbo4_pca"
  "turbo4333_pca"
  "turbo4322_pca"
  "turbo4211_pca"
)

for model in "${MODELS[@]}"; do
  stem="$(basename "$model" .gguf)"
  calib="calibration_data/${stem}_wiki_train_pca.json"
  log="experiments/turbo_pca_wiki/${stem}_calibrate.log"

  echo "=== calibrating $model ==="
  /usr/bin/time -p build/bin/llama-kv-calibrate \
    -m "$model" \
    -f wikitext-2-raw/wiki.train.raw \
    --kv-pca-calibrate \
    --kv-empvar-calibrate-out "$calib" \
    -c 512 \
    --chunks 4 \
    > "$log" 2>&1

  if [ $? -ne 0 ]; then
    echo "FAILED calibration for $model; see $log"
    exit 1
  fi

  for ct in "${CACHE_TYPES[@]}"; do
    out="experiments/turbo_pca_wiki/${stem}_${ct}_ppl.log"

    echo "=== perplexity $model $ct ==="
    GGML_METAL_TURBO_PCA_JSON_FILE="$calib" \
    /usr/bin/time -p build/bin/llama-perplexity \
      -m "$model" \
      -f wikitext-2-raw/wiki.test.raw \
      -c 512 \
      --chunks 4 \
      -ngl 99 \
      -fit off \
      -fa on \
      -ctk "$ct" \
      -ctv "$ct" \
      > "$out" 2>&1

    if [ $? -ne 0 ]; then
      echo "FAILED perplexity for $model $ct; see $out"
    fi
  done
done
