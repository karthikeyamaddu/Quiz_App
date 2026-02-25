import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_PDF_PATH = r"C:\Users\M V KARTHIKEYA\Downloads\M V Karthikeya Resume (1).pdf"
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "t5-quiz-finetune"


def extract_text_from_pdf(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        raise ValueError("No text extracted from PDF")
    return cleaned


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def normalize_question(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    q_text = str(question.get("question") or "").strip()
    options = question.get("options") or []
    options = [str(option).strip() for option in options if str(option).strip()]
    correct = str(question.get("correctAnswer") or "").strip()
    explanation = str(question.get("explanation") or "").strip()

    if not q_text or len(options) < 4:
        return None

    options = options[:4]

    if correct in ["A", "B", "C", "D"]:
        idx = ord(correct) - ord("A")
        if 0 <= idx < len(options):
            correct = options[idx]

    if not correct or correct not in options:
        correct = options[0]

    return {
        "question": q_text,
        "options": options,
        "correctAnswer": correct,
        "explanation": explanation,
    }


def parse_t5_output(raw_text: str) -> Optional[Dict[str, Any]]:
    text = raw_text.strip()
    if not text:
        return None

    json_match = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if json_match:
        text = json_match.group(1).strip()

    try:
        payload = json.loads(text)
        if isinstance(payload, list) and payload:
            payload = payload[0]
        if isinstance(payload, dict):
            normalized = normalize_question(payload)
            if normalized:
                return normalized
    except Exception:
        pass

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current = {
        "question": "",
        "options": [],
        "correctAnswer": "",
        "explanation": "",
    }
    for line in lines:
        if re.match(r"^Q[:\).\-]", line, flags=re.IGNORECASE):
            current["question"] = re.sub(r"^Q[:\).\-]\s*", "", line, flags=re.IGNORECASE)
        elif re.match(r"^[A-D][\).]", line):
            current["options"].append(re.sub(r"^[A-D][\).]\s*", "", line))
        elif line.lower().startswith("correct"):
            current["correctAnswer"] = line.split(":", 1)[-1].strip()
        elif line.lower().startswith("explanation"):
            current["explanation"] = line.split(":", 1)[-1].strip()

    normalized = normalize_question(current)
    if normalized:
        return normalized

    inline_match = re.search(
        r"Question\s*:\s*(.*?)\s*Options\s*:\s*(.*?)\s*Correct\s*Answer\s*:\s*(.*?)(?:\s*Explanation\s*:\s*(.*))?$",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not inline_match:
        return None

    question_text = inline_match.group(1).strip()
    options_blob = inline_match.group(2).strip()
    correct_blob = inline_match.group(3).strip()
    explanation_blob = (inline_match.group(4) or "").strip()

    options = [piece.strip(" -") for piece in re.split(r"\||;", options_blob) if piece.strip()]
    if len(options) < 4:
        options = [piece.strip(" -") for piece in re.split(r",", options_blob) if piece.strip()]

    if len(options) < 4:
        return None

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

    return normalize_question(
        {
            "question": question_text,
            "options": options,
            "correctAnswer": correct_answer,
            "explanation": explanation_blob,
        }
    )


def fallback_question(context: str, index: int) -> Dict[str, Any]:
    words = [word for word in re.findall(r"[A-Za-z]{4,}", context)][:20]
    if len(words) < 4:
        words = ["Concept", "Context", "Term", "Topic"]
    return {
        "question": f"What is a key term from chunk {index + 1}?",
        "options": words[:4],
        "correctAnswer": words[0],
        "explanation": f"'{words[0]}' is directly present in this chunk.",
    }


def load_t5_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"T5 model folder not found: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception as exc:
        message = str(exc)
        if "PyPreTokenizerTypeWrapper" in message:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
        elif "SentencePiece" in message or "sentencepiece" in message.lower():
            raise RuntimeError(
                "SentencePiece is missing. Install with: pip install sentencepiece==0.2.0"
            ) from exc
        else:
            raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(device)
    model.eval()
    return tokenizer, model, device


def generate_question(tokenizer, model, device: str, context: str) -> str:
    prompt = (
        "Create one MCQ from context. Return ONLY plain text using this exact format.\n"
        "Q: <question>\n"
        "A) <option A>\n"
        "B) <option B>\n"
        "C) <option C>\n"
        "D) <option D>\n"
        "Correct: <A/B/C/D>\n"
        "Explanation: <one short line>\n\n"
        f"Context: {context}"
    )

    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=240,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Run local T5 model on a PDF to generate quiz questions.")
    parser.add_argument("--pdf", default=DEFAULT_PDF_PATH, help="Absolute path to input PDF")
    parser.add_argument("--questions", type=int, default=5, help="Number of questions to generate")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    question_count = max(1, args.questions)

    print("=" * 70)
    print("T5 PDF Question Generation Test")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print(f"Model: {MODEL_PATH}")

    text = extract_text_from_pdf(pdf_path)
    print(f"Extracted characters: {len(text)}")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No chunks generated from extracted text")

    print(f"Chunks generated: {len(chunks)}")

    tokenizer, model, device = load_t5_model(MODEL_PATH)
    print(f"Device: {device}")

    print("\n" + "=" * 70)
    print("Generated Questions (Structured)")
    print("=" * 70)

    for idx in range(question_count):
        context = chunks[idx % len(chunks)]
        raw_result = generate_question(tokenizer, model, device, context)
        parsed_result = parse_t5_output(raw_result)
        if not parsed_result:
            parsed_result = fallback_question(context, idx)
            parsed_result["explanation"] = (
                parsed_result["explanation"] + " Fallback used because raw output was unparseable."
            )

        print(f"\n--- Question {idx + 1} ---")
        print(f"Q: {parsed_result['question']}")
        print(f"A) {parsed_result['options'][0]}")
        print(f"B) {parsed_result['options'][1]}")
        print(f"C) {parsed_result['options'][2]}")
        print(f"D) {parsed_result['options'][3]}")
        correct_letter = "ABCD"[parsed_result["options"].index(parsed_result["correctAnswer"])]
        print(f"Correct: {correct_letter}")
        print(f"Explanation: {parsed_result['explanation']}")
        print("Raw:", raw_result)

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)


if __name__ == "__main__":
    main()
