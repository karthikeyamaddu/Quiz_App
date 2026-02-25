"""
Run T5 inference through a Colab + ngrok endpoint.

Expected backend contract (FastAPI):
    POST /generate
    JSON body: {"text": "...", "max_length": 128}

Usage examples:
    python run_t5_colab.py --url https://abcd-12-34-56-78.ngrok-free.app --text "Generate quiz question: ..." --max-length 120

    python run_t5_colab.py --url https://abcd-12-34-56-78.ngrok-free.app --text-file ./sample_prompt.txt --max-length 150
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import requests

# Paste your latest ngrok URL here (no trailing slash preferred).
NGROK_ENDPOINT = "https://ba05-35-198-192-71.ngrok-free.app/"

# Paste your default prompt text here so `python run_t5_colab.py` works directly.
DEFAULT_TEXT = "Generate quiz question: Corrected Version:\n\nThis text is unsuitable for quiz generation in its current form due to several issues:\n\n* **Inconsistent Formatting:** The text mixes numbered sections (6.2.1, 6.2.2), seemingly from a textbook, with an unclear introduction.  This makes it hard to create concise, focused quiz questions.\n* **Lack of Clear Focus:** The paragraph covers two types of plant movement but doesn't offer a clear, simple distinction suitable for a quiz.  The description is detailed, making it difficult to extract easily testable information.\n* **Figure Reference:** The repeated \"Figure 6.4\" is problematic.  Without the actual figure, questions referencing it cannot be answered.\n* **Ambiguous Language:** Phrases like \"electrical-chemical means\" are too vague for precise quiz questions.\n\nHere's a possible corrected version suitable for quiz generation:\n\n**Plant Movement: Two Types**\n\nPlants exhibit two main types of movement:\n\n1. **Nastic Movements (Rapid Response):** These movements are rapid, independent of growth, and occur in response to a stimulus.  An example is the sensitive plant (Mimosa pudica) rapidly folding its leaves upon touch.  These movements involve changes in cell turgor pressure (water content) causing cells to swell or shrink.  Unlike animals, plants lack specialized tissues for rapid movement.\n\n2. **Tropic Movements (Growth Response):** These movements are slower and are caused by differential growth in response to a stimulus.  For example, a pea plant's tendrils grow around a support object because the side of the tendril in contact with the object grows more slowly than the opposite side.  This directional growth gives the appearance of movement.\n\nThis revised text is more concise and focused, enabling the creation of clear and unambiguous quiz questions, such as:\n\n* What type of plant movement is independent of growth?\n* Give an example of a plant that exhibits nastic movement.\n* What causes the movement in nastic movements?\n* What is the mechanism behind tropic movements?\n* Provide an example of a tropic movement."


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        return "/generate"
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Text file is empty: {path}")
    return text


def build_payload(text: str, max_length: int) -> Dict[str, Any]:
    return {
        "text": text,
        "max_length": max(1, int(max_length)),
    }


def pretty_print_response(data: Any) -> None:
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(str(data))


def call_remote_model(
    url: str,
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int,
    verify_ssl: bool,
) -> Any:
    full_url = f"{normalize_base_url(url)}{normalize_endpoint(endpoint)}"
    response = requests.post(
        full_url,
        json=payload,
        timeout=timeout,
        verify=verify_ssl,
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()

    try:
        return response.json()
    except ValueError:
        return response.text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Colab-hosted T5 model via ngrok endpoint."
    )
    parser.add_argument(
        "--url",
        default=NGROK_ENDPOINT,
        help="Base ngrok URL (if omitted, uses NGROK_ENDPOINT variable)",
    )
    parser.add_argument(
        "--endpoint",
        default="/generate",
        help="API endpoint path (default: /generate)",
    )

    text_group = parser.add_mutually_exclusive_group(required=False)
    text_group.add_argument(
        "--text",
        help="Prompt text sent as `text` (if omitted, uses DEFAULT_TEXT)",
    )
    text_group.add_argument("--text-file", help="Path to a text file for `text`")
    text_group.add_argument(
        "--raw-json",
        help="Send your own JSON payload directly (string), bypassing auto payload",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Generation max_length sent to backend (default: 128)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (use only for troubleshooting).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if not str(args.url or "").strip():
            raise ValueError("Set NGROK_ENDPOINT in the file or pass --url")

        if args.raw_json:
            payload = json.loads(args.raw_json)
        else:
            text = ""
            if args.text:
                text = args.text.strip()
            elif args.text_file:
                text = read_text_file(Path(args.text_file))
            elif DEFAULT_TEXT.strip():
                text = DEFAULT_TEXT.strip()

            if not text:
                raise ValueError(
                    "Provide one of: --text, --text-file, --raw-json, or set DEFAULT_TEXT"
                )

            payload = build_payload(text=text, max_length=args.max_length)

        result = call_remote_model(
            url=args.url,
            endpoint=args.endpoint,
            payload=payload,
            timeout=args.timeout,
            verify_ssl=not args.insecure,
        )

        print("=" * 70)
        print("Remote inference response")
        print("=" * 70)
        pretty_print_response(result)

    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}")
        if exc.response is not None:
            print("Response body:")
            print(exc.response.text)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
