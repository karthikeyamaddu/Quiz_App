# Fine-Tune Bottleneck Report (Repo Analysis)

Date: 2026-02-23

## Scope I analyzed
- Active inference/test path:
  - `current_working/backend/app.py`
  - `current_working/run_t5_doc.py`
  - `current_working/run_t5.py`
  - `current_working/frontend/src/App.jsx`
  - `current_working/FINETUNE_MODEL_STATUS.md`
- Model/config artifacts:
  - `current_working/t5-quiz-finetune/*`
  - `final_working/t5-quiz-finetune/*`
  - `rag_using_ollama/t5-quiz-finetune/*`
  - `final_working/checkpoint-324/*`
- Training clues from notebooks:
  - `colab-files/train_model.ipynb`
  - `colab-files/model_finetuning.ipynb`

## High-confidence root causes

### 1) Storage is high mainly because the repo contains duplicated models/checkpoints and full virtual environments
Measured folder sizes:

| Path | Files | Size (bytes) | Approx |
|---|---:|---:|---:|
| `current_working` | 28,771 | 2,546,586,972 | 2.37 GiB |
| `final_working` | 43,329 | 3,873,025,635 | 3.61 GiB |
| `rag_using_ollama` | 46,249 | 4,040,814,211 | 3.76 GiB |
| `current_working/venv` | 28,449 | 1,622,764,933 | 1.51 GiB |
| `final_working/quizgen_env` | 43,298 | 2,083,014,482 | 1.94 GiB |
| `rag_using_ollama/myenv` | 46,190 | 2,248,657,612 | 2.09 GiB |
| `current_working/t5-quiz-finetune` | 8 | 894,888,949 | 0.83 GiB |
| `final_working/t5-quiz-finetune` | 8 | 894,888,949 | 0.83 GiB |
| `rag_using_ollama/t5-quiz-finetune` | 20 | 1,789,800,494 | 1.67 GiB |
| `final_working/checkpoint-324` | 12 | 894,911,545 | 0.83 GiB |

What this means:
- The same fine-tuned model/checkpoint is copied in multiple folders (several GB total).
- 3 full Python environments are inside the repo, adding ~5.5 GiB by themselves.
- There is no `.gitignore` at repo root, so large artifacts are easy to accumulate.

### 2) Inference is slow because generation settings are expensive for CPU and still heavy on GPU
Evidence:
- `padding="max_length"` + `max_length=512` input encoding:
  - `current_working/backend/app.py:370`
  - `current_working/backend/app.py:372`
  - `current_working/run_t5_doc.py:210`
  - `current_working/run_t5_doc.py:211`
- Decoder settings:
  - `num_beams=4` at `current_working/backend/app.py:379` and `current_working/run_t5_doc.py:222`
  - `max_length=240` at `current_working/backend/app.py:378` and `current_working/run_t5_doc.py:221`
- Questions are generated sequentially in a loop (one generate call per question):
  - `current_working/backend/app.py:357` and `current_working/backend/app.py:362`

Impact:
- Fixed-length 512 token padding wastes compute for shorter prompts.
- Beam size 4 multiplies decode work.
- 240-token output cap is high for one MCQ.
- Sequential per-question decoding scales linearly with count and chunk count.

### 3) Runtime memory is higher than needed for inference
Evidence:
- Model config is float32 and `use_cache` is disabled:
  - `current_working/t5-quiz-finetune/config.json:56` (`"torch_dtype": "float32"`)
  - `current_working/t5-quiz-finetune/config.json:58` (`"use_cache": false`)
- Model is T5-base class size (`d_model=768`, 12 layers):
  - `current_working/t5-quiz-finetune/config.json:8`
  - `current_working/t5-quiz-finetune/config.json:20`
  - `current_working/t5-quiz-finetune/config.json:22`
- Flask runs with debug reloader:
  - `current_working/backend/app.py:833` (`debug=True`)

Impact:
- Float32 weights increase RAM/VRAM footprint.
- `use_cache=false` slows autoregressive decoding.
- Flask debug reloader can duplicate processes/model load in development.

### 4) End-to-end response time and memory also increase due very large JSON payload handling
Evidence:
- Full pipeline response returns full extracted text + all chunks + stage outputs:
  - `current_working/backend/app.py:819`
  - `current_working/backend/app.py:820`
  - `current_working/backend/app.py:821`
  - `current_working/backend/app.py:822`
  - `current_working/backend/app.py:823`
  - `current_working/backend/app.py:824`
- Finetune stage includes `rawOutputs` and full prompt/raw content:
  - `current_working/backend/app.py:331`
  - `current_working/backend/app.py:434`
- Frontend stores full payload and stage payloads simultaneously:
  - `current_working/frontend/src/App.jsx:93`
  - `current_working/frontend/src/App.jsx:94`
  - `current_working/frontend/src/App.jsx:100`
  - `current_working/frontend/src/App.jsx:106`
  - `current_working/frontend/src/App.jsx:107`
  - `current_working/frontend/src/App.jsx:108`
  - `current_working/frontend/src/App.jsx:109`
- Frontend renders full raw JSON views:
  - `current_working/frontend/src/App.jsx:499`
  - `current_working/frontend/src/App.jsx:500`

Impact:
- More serialization/deserialization time.
- Higher frontend memory due duplicated state objects and raw debug payloads.
- Perceived slowness rises even when model generation is done.

### 5) Output parse instability causes fallback usage and extra wasted inference cycles
Evidence:
- Dataset is prompt/completion style:
  - keys are `prompt` and `completion` (observed in JSONL samples).
- Dataset characteristics:
  - `train`: 433 rows, prompt chars avg ~1731, max 2642, completion avg ~269
  - `val`: 87 rows, prompt chars avg ~1656, max 2483, completion avg ~277
- Inference expects strict fixed MCQ format prompt, different from many training examples.
- Fallback path is frequently needed per status notes:
  - `current_working/FINETUNE_MODEL_STATUS.md`

Impact:
- Model often emits text not matching strict parser.
- Time is spent generating text that later gets discarded/fallbacked.

### 6) Version mismatch risk
Evidence:
- Fine-tuned config says `transformers_version: 4.50.3`:
  - `current_working/t5-quiz-finetune/config.json:57`
- Runtime requirements pin `transformers==4.35.2`:
  - `current_working/requirements.txt`

Impact:
- Potential tokenizer/model behavior differences and compatibility friction.

## Prioritized recommendations (no code changed)

## Priority 1: Biggest speed wins
1. Use lean decode defaults for finetune stage:
   - `num_beams=1`
   - cap output with `max_new_tokens` around 64-96 (instead of high `max_length`)
   - avoid `padding="max_length"` during inference; use dynamic padding only.
2. Enable decoder cache for inference (`use_cache=True` during generation/runtime config).
3. Keep `finetuneQuestionCount` low by default on CPU (1-2).
4. Add a per-question timeout/fail-fast path to fallback instead of waiting indefinitely.

## Priority 2: Memory + startup
1. Run backend without Flask debug reloader when testing model latency (`debug=False`).
2. Use lower precision where hardware supports:
   - GPU: fp16/bf16 inference.
   - CPU: dynamic quantization or int8 ONNX/OpenVINO path.
3. Reduce frontend retained payload size:
   - avoid storing `fullStage` and duplicated stage objects together for normal runs.
   - gate huge raw JSON rendering behind explicit debug mode.

## Priority 3: Storage cleanup strategy
1. Keep one canonical model directory and remove duplicated copies/checkpoints.
2. Do not keep virtualenv folders inside repo snapshots.
3. Add a root `.gitignore` covering:
   - `venv/`, `myenv/`, `quizgen_env/`
   - `checkpoint-*`
   - large model binaries (`*.safetensors`) unless intentionally versioned.
4. If model versioning is needed, move large artifacts to an external model store (Hugging Face hub or artifact storage) and pull on demand.

## Priority 4: Quality + latency together
1. Align training format with inference prompt format (strict Q/A template).
2. Shorten training prompts (many are very long); reduce irrelevant instruction noise.
3. Add a two-attempt generate strategy:
   - fast decode attempt first.
   - only second attempt if parser fails.
4. Add telemetry:
   - model load time
   - per-question encode/decode time
   - fallback rate
   - output token count

## Practical “quick win” sequence
1. Consolidate to one model folder and one environment copy (largest disk win).
2. Disable debug reloader and measure baseline latency.
3. Reduce decode settings (beams/output length/dynamic padding).
4. Trim response payload size for normal UI path.
5. Then revisit fine-tune dataset formatting for output reliability improvements.

## Expected improvements (realistic)
- Storage: several GB reduction immediately (artifact/env cleanup).
- Inference latency: often 2-5x faster from decode simplification + cache + no debug reloader.
- Memory pressure: noticeably lower from precision/runtime/payload changes.
- Output reliability: reduced fallback rate after training/inference format alignment.

