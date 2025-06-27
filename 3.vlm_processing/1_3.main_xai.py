import os
import re
import csv
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if XAI_API_KEY is None:
    raise ValueError("XAI_API_KEY not found in environment.")

API_URL = "https://api.x.ai/v1/chat/completions"

MODEL_NAME = "grok-2-vision-latest"

this_dir = Path(__file__).parent
with (this_dir / "system_prompt.txt").open(encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
with (this_dir / "user_prompt.txt").open(encoding="utf-8") as f:
    USER_PROMPT = f.read()

def extract_predicted_index(text: str) -> float:
    m = re.search(r"Ki[\s-]?67[^%]*?([0-9]+(?:\.[0-9]+)?)\s*%", text, re.I | re.S)
    if m:
        return float(m.group(1))
    # fallback: any % in the text
    perc = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    if perc:
        return float(perc[-1])
    raise ValueError("Ki‑67 value not found in model output.")

def calculate_true_index(json_path: str) -> float:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pos = sum(1 for c in data if c.get("label_id") == 1)
    neg = sum(1 for c in data if c.get("label_id") == 2)
    return round((pos / (pos + neg)) * 100, 2) if pos + neg else 0.0

def _encode_image(img_path: Path) -> Tuple[str, str]:
    mime = "jpeg" if img_path.suffix.lower() in {".jpg", ".jpeg"} else "png"
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    return mime, b64

def predict_with_grok(img_path: Path) -> Tuple[float, str]:
    mime, img_b64 = _encode_image(img_path)

    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{img_b64}"},
                    },
                ],
            },
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {XAI_API_KEY}",
    }

    resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"xAI API error {resp.status_code}: {resp.text}")

    content = resp.json()["choices"][0]["message"]["content"]
    return extract_predicted_index(content), content

def main(data_folder: str, out_parent: str | None = None) -> None:
    timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    parent = Path(out_parent).resolve() if out_parent else Path(this_dir)
    output_dir = parent / f"output_xai_{MODEL_NAME.replace('/', '_')}_date_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "ki67_log.txt"
    csv_path = output_dir / "ki67_results.csv"
    llm_path = output_dir / "llm_responses.txt"
    plot_path = output_dir / "ki67_pred_vs_true.png"

    trues, preds = [], []
    processed: set[str] = set()

    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            next(f, None)
            processed = {line.split(",")[0] for line in f if line.strip()}

    with log_path.open("a", encoding="utf-8") as logf, csv_path.open("a", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        if csv_path.stat().st_size == 0:
            writer.writerow(["image", "predicted", "true"])

        for fname in sorted(os.listdir(data_folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")) or fname in processed:
                continue

            img_path = Path(data_folder) / fname
            json_path = img_path.with_suffix(".json")
            if not json_path.is_file():
                print(f"JSON missing for {fname}")
                continue

            try:
                true_idx = calculate_true_index(json_path)
                pred_idx, full_resp = predict_with_grok(img_path)

                with llm_path.open("a", encoding="utf-8") as respf:
                    respf.write(f"\n===== {fname} =====\n{full_resp.strip()}\n")

                logf.write(f"{fname},{pred_idx:.2f},{true_idx:.2f}\n")
                writer.writerow([fname, f"{pred_idx:.2f}", f"{true_idx:.2f}"])
                trues.append(true_idx)
                preds.append(pred_idx)
                print(f"{fname}: predicted {pred_idx:.2f}  true {true_idx:.2f}")
            except Exception as e:
                print(f"Error on {fname}: {e}")

    if trues and preds:
        plt.figure(figsize=(6, 6))
        plt.scatter(trues, preds, marker="x")
        plt.plot([0, 100], [0, 100], linewidth=1, linestyle="--")
        plt.xlabel("True Ki‑67 (%)")
        plt.ylabel("Predicted Ki‑67 (%)")
        plt.title("Ki‑67 Predicted vs True (xAI)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print(
            "Usage:\n"
            "  python 3.vlm_processing/1_3.main_xai.py <processed_dataset> [<output_parent_dir>]\n"
            "Example:\n"
            "  python 3.vlm_processing/1_3.main_xai.py "
            "1.data_access/data_sample/3.data_processed "
            "5.results"
        )
        sys.exit(1)

    data_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) == 3 else None
    main(data_dir, out_dir)