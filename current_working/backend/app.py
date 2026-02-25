import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz
import requests
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import google.generativeai as genai


load_dotenv()

app = Flask(__name__)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173",
).split(",")
CORS(app, resources={r"/*": {"origins": [origin.strip() for origin in cors_origins]}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_PRIMARY = os.getenv("GEMINI_MODEL_PRIMARY", "gemini-2.5-flash")
GEMINI_MODEL_SECONDARY = os.getenv("GEMINI_MODEL_SECONDARY", "gemini-flash-latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2-uncensored")
NGROK_T5_URL = os.getenv("NGROK_T5_URL", "").rstrip("/")
NGROK_T5_ENDPOINT = os.getenv("NGROK_T5_ENDPOINT", "/generate")
NGROK_T5_MAX_LENGTH = int(os.getenv("NGROK_T5_MAX_LENGTH", "180"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

CURRENT_WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T5_MODEL_PATH = os.path.join(CURRENT_WORKING_DIR, "t5-quiz-finetune")

# Note: Local model loading is now skipped if NGROK_T5_URL is set
_TOKENIZER = None
_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PipelineConfig:
    question_mode: str = "exact"
    question_count: int = 10
    finetune_question_count: int = 2
    rag_question_count: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 150
    ollama_model: str = OLLAMA_MODEL
    gemini_primary: str = GEMINI_MODEL_PRIMARY
    gemini_secondary: str = GEMINI_MODEL_SECONDARY


def get_t5_model() -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    global _TOKENIZER, _MODEL
    if _TOKENIZER is not None and _MODEL is not None:
        return _TOKENIZER, _MODEL

    if not os.path.exists(T5_MODEL_PATH):
        raise ValueError(f"T5 model path not found: {T5_MODEL_PATH}")

    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(T5_MODEL_PATH)
    except Exception as exc:
        message = str(exc)
        if "PyPreTokenizerTypeWrapper" in message:
            _TOKENIZER = AutoTokenizer.from_pretrained(T5_MODEL_PATH, use_fast=False)
        elif "SentencePiece" in message or "sentencepiece" in message.lower():
            raise ValueError(
                "SentencePiece dependency is missing. Install with: pip install sentencepiece==0.2.0"
            ) from exc
        else:
            raise
    _MODEL = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_PATH)
    _MODEL.to(_DEVICE)
    _MODEL.eval()
    return _TOKENIZER, _MODEL


def extract_text_from_pdf(file_storage) -> str:
    try:
        pdf_document = fitz.open(stream=file_storage.read(), filetype="pdf")
        text = "\n".join(page.get_text("text") for page in pdf_document)
        pdf_document.close()
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            raise ValueError("No text extracted from PDF")
        return cleaned
    except Exception as exc:
        raise ValueError(f"Failed to extract PDF text: {exc}")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    chunks: List[Dict[str, Any]] = []
    start = 0
    index = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "index": index,
                    "start": start,
                    "end": end,
                    "length": len(chunk),
                    "text": chunk,
                }
            )
            index += 1
        start += step
    return chunks


def _extract_json_block(content: str) -> Optional[str]:
    fenced = re.search(r"```json\s*(.*?)```", content, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    generic = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
    return generic.group(1).strip() if generic else None


def _normalize_question(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(question, dict):
        return None

    q_text = (
        question.get("question")
        or question.get("q")
        or question.get("prompt")
        or ""
    ).strip()
    options = question.get("options") or question.get("choices") or []
    if isinstance(options, dict):
        options = list(options.values())
    options = [str(item).strip() for item in options if str(item).strip()]

    correct = (
        question.get("correctAnswer")
        or question.get("correct")
        or question.get("answer")
        or ""
    )
    correct = str(correct).strip()

    if len(options) < 4:
        return None

    if correct in ["A", "B", "C", "D"]:
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(options):
            correct = options[idx]

    if correct and correct not in options:
        correct = options[0]
    if not correct:
        correct = options[0]

    explanation = str(question.get("explanation") or "").strip()
    return {
        "question": q_text,
        "options": options[:4],
        "correctAnswer": correct,
        "explanation": explanation,
    }


def parse_questions_from_text(raw_text: str) -> List[Dict[str, Any]]:
    print(f"[PARSE] Parsing text, length: {len(raw_text)}")
    raw_text = raw_text.strip()
    if not raw_text:
        print(f"[PARSE] Text is empty after strip, returning []")
        return []

    json_block = _extract_json_block(raw_text)
    if json_block:
        print(f"[PARSE] Found JSON block, length: {len(json_block)}")
        try:
            parsed = json.loads(json_block)
            print(f"[PARSE] Successfully parsed JSON, type: {type(parsed).__name__}")
            if isinstance(parsed, dict):
                parsed = parsed.get("questions", [])
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    norm = _normalize_question(item)
                    if norm and norm["question"]:
                        result.append(norm)
                if result:
                    print(f"[PARSE] ✓ JSON parse successful: {len(result)} questions extracted")
                    return result
        except Exception as e:
            print(f"[PARSE] JSON parse failed: {e}")
            pass
    else:
        print(f"[PARSE] No JSON block found")

    print(f"[PARSE] Attempting line-by-line format parsing")
    questions = []
    current: Optional[Dict[str, Any]] = None
    line_count = 0
    for line in raw_text.splitlines():
        text = line.strip()
        line_count += 1
        if not text:
            continue
        if re.match(r"^Q[:\).\-]", text, flags=re.IGNORECASE):
            if current:
                norm = _normalize_question(current)
                if norm and norm["question"]:
                    questions.append(norm)
            current = {
                "question": re.sub(r"^Q[:\).\-]\s*", "", text, flags=re.IGNORECASE),
                "options": [],
                "correctAnswer": "",
                "explanation": "",
            }
        elif re.match(r"^[A-D][\).]", text):
            if current is not None:
                current["options"].append(re.sub(r"^[A-D][\).]\s*", "", text))
        elif text.lower().startswith("correct"):
            if current is not None:
                value = text.split(":", 1)[-1].strip()
                if value in ["A", "B", "C", "D"]:
                    idx = ord(value) - ord("A")
                    if idx < len(current["options"]):
                        current["correctAnswer"] = current["options"][idx]
                else:
                    current["correctAnswer"] = value
        elif text.lower().startswith("explanation"):
            if current is not None:
                current["explanation"] = text.split(":", 1)[-1].strip()

    if current:
        norm = _normalize_question(current)
        if norm and norm["question"]:
            questions.append(norm)

    if questions:
        print(f"[PARSE] ✓ Line-by-line parse successful: {len(questions)} questions found")
        return questions
    else:
        print(f"[PARSE] Line-by-line parse found no questions (scanned {line_count} lines)")

    inline_match = re.search(
        r"Question\s*:\s*(.*?)\s*Options\s*:\s*(.*?)\s*Correct\s*Answer\s*:\s*(.*?)(?:\s*Explanation\s*:\s*(.*))?$",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if inline_match:
        question_text = inline_match.group(1).strip()
        options_blob = inline_match.group(2).strip()
        correct_blob = inline_match.group(3).strip()
        explanation_blob = (inline_match.group(4) or "").strip()

        options = [piece.strip(" -") for piece in re.split(r"\||;", options_blob) if piece.strip()]
        if len(options) < 4:
            options = [piece.strip(" -") for piece in re.split(r",", options_blob) if piece.strip()]

        if len(options) >= 4:
            options = options[:4]
            correct_answer = ""

            if re.fullmatch(r"\d+", correct_blob):
                idx = int(correct_blob) - 1
                if 0 <= idx < len(options):
                    correct_answer = options[idx]
            elif correct_blob.upper() in ["A", "B", "C", "D"]:
                idx = ord(correct_blob.upper()) - ord("A")
                if 0 <= idx < len(options):
                    correct_answer = options[idx]
            else:
                for option in options:
                    if correct_blob.lower() in option.lower() or option.lower() in correct_blob.lower():
                        correct_answer = option
                        break

            if not correct_answer:
                correct_answer = options[0]

            normalized = _normalize_question(
                {
                    "question": question_text,
                    "options": options,
                    "correctAnswer": correct_answer,
                    "explanation": explanation_blob,
                }
            )
            if normalized and normalized["question"]:
                return [normalized]

    return questions


def fallback_question_from_chunk(chunk_text_value: str, index: int) -> Dict[str, Any]:
    words = [w for w in re.findall(r"[A-Za-z]{4,}", chunk_text_value)][:20]
    if len(words) < 4:
        words = ["Concept", "Context", "Term", "Topic"]
    correct = words[0]
    options = words[:4]
    return {
        "question": f"What is a key term in chunk {index + 1}?",
        "options": options,
        "correctAnswer": correct,
        "explanation": f"'{correct}' appears in the source chunk and is contextually relevant.",
    }


def build_finetune_fallback_result(
    chunks: List[Dict[str, Any]], question_count: int, reason: str
) -> Dict[str, Any]:
    questions = []
    raw_outputs = []
    safe_count = max(1, question_count)
    for index in range(safe_count):
        chunk = chunks[index % len(chunks)]
        fallback = fallback_question_from_chunk(chunk.get("text", ""), index)
        fallback["source"] = "finetuned_t5"
        fallback["chunkIndex"] = chunk.get("index", index)
        fallback["usedFallback"] = True
        fallback["fallbackReason"] = reason
        fallback["modelOutput"] = ""
        questions.append(fallback)
        raw_outputs.append(
            {
                "questionIndex": index,
                "chunkIndex": chunk.get("index", index),
                "raw": "",
                "usedFallback": True,
                "fallbackReason": reason,
                "prompt": "",
            }
        )

    return {
        "questions": questions,
        "rawOutputs": raw_outputs,
        "warning": reason,
    }


def compute_question_count(text: str, mode: str, requested_count: int) -> int:
    if mode == "auto":
        estimated = max(3, len(text.split()) // 120)
        return min(25, estimated)
    return max(1, min(50, requested_count))


def build_t5_prompt(context_text: str) -> str:
    return (
        "Create one MCQ from context. Return ONLY plain text using this exact format.\n"
        "Q: <question>\n"
        "A) <option A>\n"
        "B) <option B>\n"
        "C) <option C>\n"
        "D) <option D>\n"
        "Correct: <A/B/C/D>\n"
        "Explanation: <one short line>\n\n"
        f"Context: {context_text}"
    )


def run_finetuned_stage(chunks: List[Dict[str, Any]], question_count: int) -> Dict[str, Any]:
    print("\n" + "="*80)
    print("[FINETUNE STAGE] Starting remote T5 question generation")
    print(f"[FINETUNE STAGE] Question count requested: {question_count}")
    print(f"[FINETUNE STAGE] Chunks available: {len(chunks)}")
    
    if not NGROK_T5_URL:
        print("[FINETUNE STAGE] ERROR: NGROK_T5_URL not set!")
        raise ValueError("NGROK_T5_URL environment variable not set. Set it to your Colab ngrok endpoint.")

    raw_outputs = []
    questions = []
    endpoint_url = f"{NGROK_T5_URL}{NGROK_T5_ENDPOINT if NGROK_T5_ENDPOINT.startswith('/') else f'/{NGROK_T5_ENDPOINT}'}"
    print(f"[FINETUNE STAGE] Remote endpoint: {endpoint_url}")
    print(f"[FINETUNE STAGE] Max length: {NGROK_T5_MAX_LENGTH}")

    for index in range(question_count):
        print(f"\n[FINETUNE Q{index+1}] Processing question {index+1}/{question_count}")
        chunk = chunks[index % len(chunks)]
        print(f"[FINETUNE Q{index+1}] Using chunk {chunk['index']}, length: {chunk['length']} chars")
        
        prompt = build_t5_prompt(chunk["text"])
        print(f"[FINETUNE Q{index+1}] Prompt built, length: {len(prompt)} chars")
        print(f"[FINETUNE Q{index+1}] Prompt preview: {prompt[:200]}...")

        try:
            payload = {"text": prompt, "max_length": NGROK_T5_MAX_LENGTH}
            print(f"[FINETUNE Q{index+1}] Sending request to {endpoint_url}...")
            response = requests.post(endpoint_url, json=payload, timeout=180)
            print(f"[FINETUNE Q{index+1}] Response status: {response.status_code}")
            
            response.raise_for_status()
            result = response.json()
            print(f"[FINETUNE Q{index+1}] Response JSON keys: {list(result.keys())}")
            
            text_out = str(result.get("output", "")).strip()
            print(f"[FINETUNE Q{index+1}] Model output length: {len(text_out)} chars")
            print(f"[FINETUNE Q{index+1}] Model output: {text_out[:300]}...")

            used_fallback = False
            fallback_reason = ""
            print(f"[FINETUNE Q{index+1}] Attempting to parse output...")
            parsed = parse_questions_from_text(text_out)
            print(f"[FINETUNE Q{index+1}] Parse result: {len(parsed) if parsed else 0} questions found")
            
            if parsed:
                selected = parsed[0]
                print(f"[FINETUNE Q{index+1}] ✓ Successfully parsed question: '{selected.get('question', '')[:100]}'")
            else:
                print(f"[FINETUNE Q{index+1}] ✗ Could not parse, using fallback")
                selected = fallback_question_from_chunk(chunk["text"], index)
                selected["explanation"] = (
                    selected.get("explanation", "") + " Generated fallback due to unparseable remote T5 output."
                ).strip()
                used_fallback = True
                fallback_reason = "Unparseable remote model output"
                print(f"[FINETUNE Q{index+1}] Fallback question: '{selected.get('question', '')}'")

            raw_outputs.append(
                {
                    "questionIndex": index,
                    "chunkIndex": chunk["index"],
                    "raw": text_out,
                    "usedFallback": used_fallback,
                    "fallbackReason": fallback_reason,
                    "prompt": prompt,
                }
            )
        except Exception as exc:
            print(f"[FINETUNE Q{index+1}] ✗ ERROR: {type(exc).__name__}: {exc}")
            used_fallback = True
            fallback_reason = f"Remote T5 generation error: {exc}"
            raw_outputs.append(
                {
                    "questionIndex": index,
                    "chunkIndex": chunk["index"],
                    "raw": "",
                    "usedFallback": used_fallback,
                    "fallbackReason": fallback_reason,
                    "prompt": prompt,
                }
            )
            selected = fallback_question_from_chunk(chunk["text"], index)
            selected["explanation"] = (
                selected.get("explanation", "") + f" Generated fallback due to remote error: {exc}"
            ).strip()
            print(f"[FINETUNE Q{index+1}] Using fallback: '{selected.get('question', '')}'")

        selected["source"] = "finetuned_t5_remote"
        selected["chunkIndex"] = chunk["index"]
        selected["usedFallback"] = used_fallback
        selected["fallbackReason"] = fallback_reason
        selected["modelOutput"] = raw_outputs[-1].get("raw", "")
        questions.append(selected)
        print(f"[FINETUNE Q{index+1}] Final: fallback={used_fallback}, source={selected['source']}")

    return {
        "questions": questions,
        "rawOutputs": raw_outputs,
    }


def run_ollama_validation_stage(
    chunks: List[Dict[str, Any]], questions: List[Dict[str, Any]], ollama_model: str, question_count: int = None
) -> Dict[str, Any]:
    validated = []
    raw_responses = []
    context = "\n\n".join(chunk["text"] for chunk in chunks[:4])
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    
    # Limit number of questions to validate if specified
    questions_to_validate = questions[:question_count] if question_count else questions

    for idx, question in enumerate(questions_to_validate):
        prompt = (
            "You are validating an MCQ against source context. "
            "Return ONLY JSON with keys: verdict(valid|invalid), score(0-100), reason, correctedQuestion(optional), "
            "correctedOptions(optional), correctedAnswer(optional), correctedExplanation(optional).\n\n"
            f"Context:\n{context}\n\nQuestion JSON:\n{json.dumps(question, ensure_ascii=False)}"
        )
        try:
            response = requests.post(
                endpoint,
                json={"model": ollama_model, "prompt": prompt, "stream": False},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload.get("response", "")
            raw_responses.append({"questionIndex": idx, "raw": content})

            parsed_json = _extract_json_block(content) or content
            parsed = json.loads(parsed_json)
            verdict = str(parsed.get("verdict", "invalid")).lower()
            score = float(parsed.get("score", 0))
            is_valid = verdict == "valid" and score >= 50

            final_question = {
                "question": parsed.get("correctedQuestion") or question["question"],
                "options": parsed.get("correctedOptions") or question["options"],
                "correctAnswer": parsed.get("correctedAnswer") or question["correctAnswer"],
                "explanation": parsed.get("correctedExplanation") or question.get("explanation", ""),
                "source": "rag_validated",
                "rag": {
                    "verdict": verdict,
                    "score": score,
                    "reason": parsed.get("reason", ""),
                },
            }
            validated.append(final_question)
            if not is_valid:
                final_question["rag"]["warning"] = "Low confidence validation"
        except Exception as exc:
            raw_responses.append({"questionIndex": idx, "raw": f"RAG error: {exc}"})
            fallback = dict(question)
            fallback["source"] = "rag_fallback"
            fallback["rag"] = {
                "verdict": "invalid",
                "score": 0,
                "reason": f"Ollama validation failed: {exc}",
                "warning": "Used original question due to RAG failure",
            }
            validated.append(fallback)

    return {"questions": validated, "rawOutputs": raw_responses}


def run_primary_gemini_stage(text: str, question_count: int, model_name: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {
            "questions": [],
            "rawOutput": "GEMINI_API_KEY missing. Primary Gemini stage skipped.",
            "warning": "Gemini unavailable",
        }

    prompt = (
        "Generate multiple-choice quiz questions from the text. "
        f"Generate exactly {question_count} questions. "
        "Return only JSON array, each item containing: question, options(4), correctAnswer, explanation.\n\n"
        f"Text:\n{text[:12000]}"
    )
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    raw = response.text if response and getattr(response, "text", None) else ""
    questions = parse_questions_from_text(raw)

    if len(questions) < 1:
        return {
            "questions": [],
            "rawOutput": raw,
            "warning": "Primary Gemini returned unparseable output",
        }

    for item in questions:
        item["source"] = "gemini_primary"
    return {
        "questions": questions[:question_count],
        "rawOutput": raw,
    }


def run_secondary_gemini_stage(
    rag_questions: List[Dict[str, Any]],
    primary_questions: List[Dict[str, Any]],
    question_count: int,
    model_name: str,
) -> Dict[str, Any]:
    candidates = rag_questions + primary_questions
    if not candidates:
        return {
            "evaluated": [],
            "finalQuiz": [],
            "rawOutput": "No candidate questions found for secondary Gemini stage",
        }

    if not GEMINI_API_KEY:
        scored = []
        for question in candidates:
            base = 65
            rag_bonus = question.get("rag", {}).get("score", 0) * 0.2
            score = min(100, round(base + rag_bonus, 1))
            item = dict(question)
            item["qualityScore"] = score
            scored.append(item)
        scored.sort(key=lambda x: x["qualityScore"], reverse=True)
        return {
            "evaluated": scored,
            "finalQuiz": scored[:question_count],
            "rawOutput": "GEMINI_API_KEY missing. Used local scoring fallback.",
            "warning": "Gemini unavailable",
        }

    prompt = (
        "Evaluate candidate MCQs and choose the best quiz set. "
        "Return ONLY JSON with keys: evaluated (array) and finalQuiz (array). "
        "Each evaluated item includes: question, options, correctAnswer, explanation, source, qualityScore(0-100), review. "
        f"Choose exactly {question_count} items in finalQuiz.\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    raw = response.text if response and getattr(response, "text", None) else ""

    try:
        block = _extract_json_block(raw) or raw
        payload = json.loads(block)
        evaluated = payload.get("evaluated", [])
        final_quiz = payload.get("finalQuiz", [])

        normalized_eval = []
        for item in evaluated:
            norm = _normalize_question(item)
            if not norm:
                continue
            norm["source"] = item.get("source", "gemini_secondary")
            norm["qualityScore"] = float(item.get("qualityScore", 0))
            norm["review"] = item.get("review", "")
            normalized_eval.append(norm)

        normalized_final = []
        for item in final_quiz:
            norm = _normalize_question(item)
            if not norm:
                continue
            norm["source"] = item.get("source", "gemini_secondary")
            norm["qualityScore"] = float(item.get("qualityScore", 0))
            normalized_final.append(norm)

        if not normalized_final:
            normalized_eval.sort(key=lambda x: x.get("qualityScore", 0), reverse=True)
            normalized_final = normalized_eval[:question_count]

        return {
            "evaluated": normalized_eval,
            "finalQuiz": normalized_final[:question_count],
            "rawOutput": raw,
        }
    except Exception as exc:
        return {
            "evaluated": [],
            "finalQuiz": candidates[:question_count],
            "rawOutput": raw,
            "warning": f"Secondary Gemini parse failure: {exc}",
        }


def parse_pipeline_config_from_request() -> PipelineConfig:
    source = request.form if request.form else (request.get_json(silent=True) or {})
    question_mode = str(source.get("questionMode", "exact")).strip().lower()
    question_count = int(source.get("questionCount", 10))
    finetune_question_count = int(source.get("finetuneQuestionCount", 2))
    rag_question_count = int(source.get("ragQuestionCount", 5))
    chunk_size = int(source.get("chunkSize", 800))
    chunk_overlap = int(source.get("chunkOverlap", 150))
    ollama_model = str(source.get("ollamaModel", OLLAMA_MODEL)).strip()
    gemini_primary = str(source.get("geminiModelPrimary", GEMINI_MODEL_PRIMARY)).strip()
    gemini_secondary = str(source.get("geminiModelSecondary", GEMINI_MODEL_SECONDARY)).strip()

    return PipelineConfig(
        question_mode=question_mode,
        question_count=question_count,
        finetune_question_count=max(1, min(50, finetune_question_count)),
        rag_question_count=max(1, min(50, rag_question_count)),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ollama_model=ollama_model,
        gemini_primary=gemini_primary,
        gemini_secondary=gemini_secondary,
    )


@app.route("/api/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "device": _DEVICE,
            "t5ModelPath": T5_MODEL_PATH,
            "geminiConfigured": bool(GEMINI_API_KEY),
            "ollamaBaseUrl": OLLAMA_BASE_URL,
            "defaults": {
                "primaryGemini": GEMINI_MODEL_PRIMARY,
                "secondaryGemini": GEMINI_MODEL_SECONDARY,
                "ollamaModel": OLLAMA_MODEL,
                "chunkSize": 800,
                "chunkOverlap": 150,
            },
        }
    )


@app.route("/api/stage/extract", methods=["POST"])
def stage_extract() -> Any:
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        text = extract_text_from_pdf(file)
        return jsonify(
            {
                "filename": file.filename,
                "text": text,
                "textLength": len(text),
                "wordCount": len(text.split()),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stage/chunk", methods=["POST"])
def stage_chunk() -> Any:
    try:
        payload = request.get_json(force=True)
        text = payload.get("text", "")
        chunk_size = int(payload.get("chunkSize", 800))
        chunk_overlap = int(payload.get("chunkOverlap", 150))
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        print(f"[CHUNK STAGE] ✓ Chunking complete: {len(chunks)} chunks created")
        for chunk in chunks:
            print(f"[CHUNK STAGE]   Chunk {chunk['index']}: {chunk['length']} chars")
        return jsonify(
            {
                "count": len(chunks),
                "chunkSize": chunk_size,
                "chunkOverlap": chunk_overlap,
                "chunks": chunks,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stage/finetune", methods=["POST"])
def stage_finetune() -> Any:
    print("\n" + "="*80)
    print("[FINETUNE ENDPOINT] Request received at /api/stage/finetune")
    try:
        payload = request.get_json(force=True)
        chunks = payload.get("chunks", [])
        question_count = int(payload.get("questionCount", 10))
        print(f"[FINETUNE ENDPOINT] Chunks: {len(chunks)}, Questions: {question_count}")
        
        if not chunks:
            return jsonify({"error": "chunks are required"}), 400
        try:
            result = run_finetuned_stage(chunks, question_count)
            qcount = len(result.get('questions', []))
            fbcount = sum(1 for q in result.get('questions', []) if q.get('usedFallback'))
            print(f"[FINETUNE ENDPOINT] Complete: {qcount} questions, {fbcount} fallbacks")
        except Exception as exc:
            print(f"[FINETUNE ENDPOINT] ERROR: {exc}")
            result = build_finetune_fallback_result(
                chunks=chunks,
                question_count=question_count,
                reason=f"Finetune error: {exc}",
            )
        return jsonify(result)
    except Exception as exc:
        print(f"[FINETUNE ENDPOINT] EXCEPTION: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stage/rag", methods=["POST"])
def stage_rag() -> Any:
    try:
        payload = request.get_json(force=True)
        chunks = payload.get("chunks", [])
        questions = payload.get("questions", [])
        rag_question_count = payload.get("ragQuestionCount", None)
        ollama_model = payload.get("ollamaModel", OLLAMA_MODEL)
        if not chunks or not questions:
            return jsonify({"error": "chunks and questions are required"}), 400
        if rag_question_count:
            rag_question_count = int(rag_question_count)
        result = run_ollama_validation_stage(chunks, questions, ollama_model, rag_question_count)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stage/gemini-primary", methods=["POST"])
def stage_gemini_primary() -> Any:
    try:
        payload = request.get_json(force=True)
        text = payload.get("text", "")
        question_count = int(payload.get("questionCount", 10))
        model_name = payload.get("geminiModelPrimary", GEMINI_MODEL_PRIMARY)
        result = run_primary_gemini_stage(text, question_count, model_name)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/stage/gemini-secondary", methods=["POST"])
def stage_gemini_secondary() -> Any:
    try:
        payload = request.get_json(force=True)
        rag_questions = payload.get("ragQuestions", [])
        primary_questions = payload.get("primaryQuestions", [])
        question_count = int(payload.get("questionCount", 10))
        model_name = payload.get("geminiModelSecondary", GEMINI_MODEL_SECONDARY)
        result = run_secondary_gemini_stage(
            rag_questions=rag_questions,
            primary_questions=primary_questions,
            question_count=question_count,
            model_name=model_name,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/pipeline/full", methods=["POST"])
def run_full_pipeline() -> Any:
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported"}), 400

        config = parse_pipeline_config_from_request()
        text = extract_text_from_pdf(file)
        question_count = compute_question_count(text, config.question_mode, config.question_count)
        chunks = chunk_text(text, config.chunk_size, config.chunk_overlap)
        try:
            finetune_result = run_finetuned_stage(chunks, config.finetune_question_count)
        except Exception as exc:
            finetune_result = build_finetune_fallback_result(
                chunks=chunks,
                question_count=config.finetune_question_count,
                reason=f"Finetune model unavailable, fallback used: {exc}",
            )
        rag_result = run_ollama_validation_stage(chunks, finetune_result["questions"], config.ollama_model, config.rag_question_count)
        primary_result = run_primary_gemini_stage(text, question_count, config.gemini_primary)
        secondary_result = run_secondary_gemini_stage(
            rag_questions=rag_result.get("questions", []),
            primary_questions=primary_result.get("questions", []),
            question_count=question_count,
            model_name=config.gemini_secondary,
        )

        debug_payload = {
            "config": {
                "questionMode": config.question_mode,
                "questionCount": question_count,
                "finetuneQuestionCount": config.finetune_question_count,
                "ragQuestionCount": config.rag_question_count,
                "chunkSize": config.chunk_size,
                "chunkOverlap": config.chunk_overlap,
                "ollamaModel": config.ollama_model,
                "geminiModelPrimary": config.gemini_primary,
                "geminiModelSecondary": config.gemini_secondary,
            },
            "input": {
                "filename": file.filename,
                "textLength": len(text),
                "wordCount": len(text.split()),
            },
            "extractedText": text,
            "chunks": chunks,
            "finetune": finetune_result,
            "rag": rag_result,
            "primaryGemini": primary_result,
            "secondaryGemini": secondary_result,
            "finalQuiz": secondary_result.get("finalQuiz", []),
        }
        return jsonify(debug_payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
