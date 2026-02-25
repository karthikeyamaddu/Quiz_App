# Finetuned T5 Model Status (Current Working Notes)

## 1) What this finetuned model is
- **Model type:** Local finetuned T5-style seq2seq model in `current_working/t5-quiz-finetune`.
- **Tokenizer assets:** `spiece.model`, `tokenizer.json`, `tokenizer_config.json`.
- **Weights:** `model.safetensors`.
- **Usage goal:** Generate quiz-style MCQs from extracted PDF chunks.

## 2) What it does
- Takes chunked PDF text as context.
- Tries to generate **one MCQ per call** (question + 4 options + correct answer + explanation).
- Output is parsed and normalized into app quiz format.
- If output is malformed/unusable, system returns a **fallback question** so pipeline does not crash.

## 3) How it works in the app now
### Backend flow
- File: `current_working/backend/app.py`
- Endpoint: `POST /api/stage/finetune`
- Endpoint: `POST /api/pipeline/full` (finetune stage inside full run)

### Finetune stage logic (current)
1. Loads model/tokenizer from `current_working/t5-quiz-finetune`.
2. Handles tokenizer compatibility fallbacks:
   - `PyPreTokenizerTypeWrapper` -> retries with `use_fast=False`.
   - Missing SentencePiece -> clear dependency error.
3. Builds prompt from chunk text.
4. Runs generation.
5. Parses output into structured MCQ.
6. If parse fails or generation fails -> returns fallback MCQ + fallback reason.
7. Returns `questions` + `rawOutputs` (includes prompt/raw/fallback flags).

### Frontend support (current)
- File: `current_working/frontend/src/App.jsx`
- Manual stage has **Finetune button**.
- Added **Finetune Questions** control (separate from overall question count).
- Finetune panel shows:
  - structured questions
  - fallback flags/reasons
  - actual raw model outputs
  - prompt and raw output in expandable blocks
  - raw JSON includes fallback metadata

## 4) How it is currently working (observed)
- End-to-end model loading works.
- PDF -> chunk -> finetune request path works.
- Finetune no longer hard-crashes on tokenizer/model formatting errors.
- **Quality is mixed:** model sometimes emits non-strict format text.
- Parser + fallback currently keep the stage alive and return usable data.

## 5) Bottlenecks and technical issues

### A) Main bottleneck: CPU generation latency
- T5 generation is heavy on CPU with current decode settings.
- Finetune requests can appear stuck at “Running...” for minutes when question count > 1.
- Root cause: expensive decoding configuration + sequential question generation.

### B) Output quality/format instability
- Model sometimes returns inline/unstructured text instead of strict MCQ format.
- Parser handles many cases, but fallback is still frequently triggered.

### C) Operational risk
- Large chunk count + high finetune question count multiplies latency.
- User experience suffers without progress feedback/timeout for long finetune runs.

## 6) Issues currently being faced
- Finetune button may take too long (perceived freeze) on CPU.
- Inconsistent model formatting leads to fallback usage.
- Need controlled finetune question count to avoid CPU overload (partially solved with separate finetune count input).

## 7) Fix plan for next phase (recommended)

### Priority 1 (performance, must-do)
1. **Use lighter decode config** for finetune API path:
   - reduce input max length (avoid aggressive `padding=max_length` when possible)
   - use `num_beams=1`
   - set `max_new_tokens` lower (e.g., 48–96)
2. Add **hard per-question time budget** / fail-fast fallback if generation is too slow.
3. Keep default `finetuneQuestionCount` low (2) for CPU devices.

### Priority 2 (stability)
4. Add multi-attempt strategy per question:
   - try 1-2 short decode attempts
   - accept first parseable output
   - fallback only if all attempts fail
5. Improve parser coverage for common malformed formats seen in raw outputs.

### Priority 3 (UX/debug)
6. Add frontend progress indicators for finetune stage (question x/y).
7. Show latency metrics in JSON (`loadTimeMs`, `genTimeMsPerQuestion`).

## 8) Environment/Dependency notes
- Required for tokenizer fallback path:
  - `sentencepiece==0.2.0`
- Current model path is forced to:
  - `current_working/t5-quiz-finetune`
  - so working directory differences do not break model loading.

## 9) Quick command references
- Test local script:
  - `python run_t5_doc.py --pdf "C:/Users/M V KARTHIKEYA/Downloads/M V Karthikeya Resume (1).pdf" --questions 1`
- Backend run:
  - `python backend/app.py`
- Frontend run:
  - `cd frontend && npm run dev`

---

## Short summary
The finetuned T5 pipeline is **functionally integrated and resilient** (fallback-safe), but **CPU decode latency and output-format instability** are the current blockers. Next work should focus on lighter decoding, bounded runtime, and better parse-first strategies before fallback.
